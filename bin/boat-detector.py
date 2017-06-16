import numpy as np
import geojson
import subprocess, os
import protogen
import os
import shutil
import json

from osgeo import gdal
from scipy.misc import imresize
from keras.models import load_model
from gbdx_task_interface import GbdxTaskInterface
from os.path import join

class BoatDetector(GbdxTaskInterface):
    '''
    Deploys a trained CNN classifier on protogen-generated candidate regions to determine which ones contain boats.
    '''

    def __init__(self):

        GbdxTaskInterface.__init__(self)

        # Image inputs
        self.ms_dir = self.get_input_data_port('ms_image')
        self.ps_dir = self.get_input_data_port('ps_image')
        self.mask_dir = self.get_input_data_port('mask')

        # Point to ms. If there are multiple tif's in multiple subdirectories, pick one.
        self.ms_image_path, self.ms_image = [(dp, f) for dp, dn, fn in os.walk(self.ms_dir) for f in fn if 'tif' in f][0]

        # Point to ps. If there are multiple tif's in multiple subdirectories, pick one.
        self.ps_image_path, self.ps_image = [(dp, f) for dp, dn, fn in os.walk(self.ps_dir) for f in fn if 'tif' in f][0]

        # Point to mask file if it's there. If there are multiple tif's in multiple subdirectories, pick one.
        try:
            self.mask_path, self.mask = [(dp, f) for dp, dn, fn in os.walk(self.mask_dir) for f in fn if 'tif' in f][0]
        except IndexError:
            self.mask_path, self.mask = None, None

        # String inputs
        self.threshold = float(self.get_input_string_port('threshold', '0.5'))
        self.with_mask = self.get_input_string_port('with_mask', 'true')
        self.dilation = int(self.get_input_string_port('dilation', '100'))
        self.min_linearity = float(self.get_input_string_port('min_linearity', '2.0'))
        self.max_linearity = float(self.get_input_string_port('max_linearity', '8.0'))
        self.min_size = int(self.get_input_string_port('min_size', '1000'))
        self.max_size = int(self.get_input_string_port('max_size', '10000'))

        if self.with_mask in ['True', 'true', 't']:
            self.with_mask = True
        else:
            self.with_mask = False

        # Create output directory
        self.output_dir = self.get_output_data_port('results')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def extract_candidates(self):
        '''
        Use protogen to generate candidate bounding boxes.
        The function returns the name of the geojson file containing the bounding
        boxes.
        '''

        # Make the multispectral image directory the working directory
        os.chdir(self.ms_image_path)

        # Get number of bands (depends on sensor)
        no_bands = gdal.Open(self.ms_image).RasterCount

        # If mask is not provided and with_mask is true then make one
        if not self.mask and self.with_mask:

            print 'Creating water mask'
            make_mask = True

            # Compute normalized difference water index
            rbr = protogen.Interface('radex_scalar', 'band_ratio')
            rbr.radex_scalar.band_ratio.index_formula = 'IDX1'
            rbr.radex_scalar.band_ratio.output_datatype = 'UINT8'
            rbr.image = self.ms_image
            rbr.image_config.bands = [1, no_bands]
            rbr.execute()

            # Create mask
            mot = protogen.Interface('morphology', 'threshold')
            mot.morphology.threshold.algorithm = 'brute_force'
            mot.morphology.threshold.threshold = 128
            mot.morphology.threshold.new_min_value = 0
            mot.morphology.threshold.new_max_value = 255
            mot.image = rbr.output
            mot.image_config.bands = [1]
            mot.execute()

            # Remove holes (due to water bodies or shadows) from land with union find size filtering
            uff = protogen.Interface('union_find', 'filter')
            uff.unionfind.filter.labels_type = 'binary'
            uff.unionfind.filter.object_representation = 'coverage'
            uff.unionfind.filter.spatial_connectivity = 4
            uff.athos.dimensions = 2
            uff.athos.tree_type = 'union_find'
            uff.athos.area.usage = ['remove if less']
            uff.athos.area.min = [250000]
            uff.image = mot.output
            uff.image_config.bands = [1]
            uff.execute()

            mask = uff.output

            # Copy mask to output folder (for debugging purposes)
            shutil.copy(mask, self.output_dir)

        # If mask is provided and with_mask is true then use the provided one
        elif self.mask and self.with_mask:
            make_mask = False
            shutil.copy(os.path.join(self.mask_path, self.mask), '.')

        # Compute band dissimilarity map with radex
        print 'Compute dissimilarity map'
        rbd = protogen.Interface('radex_scalar', 'band_dissimilarity')
        rbd.radex_scalar.band_dissimilarity.type = 'max'
        rbd.radex_scalar.band_dissimilarity.threshold = 1
        rbd.image = self.ms_image
        rbd.image_config.bands = range(1, no_bands+1)
        rbd.execute()

        if self.with_mask:

            # Apply a dilation to remove holes in the water mask (from boats and other anomalies) and invade the coastline
            print 'Dilate water mask'
            msd = protogen.Interface('morphology', 'structural')
            msd.morphology.structural.operator = 'dilation'
            msd.morphology.structural.structuring_element = 'disk'
            msd.morphology.structural.radius1 = self.dilation
            msd.image = self.mask
            msd.image_config.bands = [1]
            msd.execute()

            if not make_mask:
                print 'Match mask and image'
                # Match the dimensions of the external mask to the dissimilarity map
                # (in case there are minor differences which will mess up the masking)
                ipm = protogen.Interface('image_processor', 'match')
                ipm.image = msd.output
                ipm.image_config.bands = [1]
                ipm.slave = rbd.output
                ipm.slave_config.bands = [1]
                ipm.execute()

            # Apply the mask on the dissimilarity map
            print 'Apply mask on dissimilarity map'
            im = protogen.Interface('image_processor', 'masking')
            im.image_processor.masking.method = 'inclusion'
            im.image_processor.masking.tree_type = 'raw'
            im.image = rbd.output
            im.image_config.bands = [1]
            if not make_mask:
                im.mask = ipm.output
            else:
                im.mask = msd.output
            im.mask_config.bands = [1]
            im.execute()

        # Min-tree filtering to find objects that conform to boat geometric characteristics
        print 'Find boat candidates with min-tree filtering'
        mf = protogen.Interface('max_tree', 'filter')
        mf.maxtree.filter.filtering_rule = 'subtractive'
        mf.maxtree.filter.spatial_connectivity = 4
        mf.maxtree.filter.tree_type = 'min_tree'
        mf.athos.dimensions = 2
        mf.athos.tree_type = 'max_tree'
        mf.athos.area.usage = ['remove if outside']
        mf.athos.area.min = [self.min_size]
        mf.athos.area.max = [self.max_size]
        mf.athos.linearity2.usage = ['remove if outside']
        mf.athos.linearity2.min = [self.min_linearity]
        mf.athos.linearity2.max = [self.max_linearity]
        if self.with_mask:
            mf.image = im.output
        else:
            mf.image = rbd.output
        mf.image_config.bands = [1]
        mf.execute()

        # Produce binary image with thresholding
        print 'Threshold min-tree output'
        mot = protogen.Interface('morphology', 'threshold')
        mot.morphology.threshold.algorithm = 'brute_force'
        mot.morphology.threshold.threshold = 100
        mot.morphology.threshold.new_min_value = 0
        mot.morphology.threshold.new_max_value = 255
        mot.image = mf.output
        mot.image_config.bands = [1]
        mot.execute()

        # Generate bounding boxes
        print 'Generate vectors'
        vbb = protogen.Interface('vectorizer', 'bounding_box')
        vbb.vectorizer.bounding_box.filetype = 'geojson'
        vbb.athos.tree_type = 'union-find'
        vbb.athos.dimensions = 2
        vbb.athos.area.export = [1]
        vbb.image = mot.output
        vbb.image_config.bands = [1]
        vbb.execute()

        # Rename geojson and copy it to output folder (for debugging purposes)
        shutil.move(vbb.output, 'candidates.geojson')
        shutil.copy('candidates.geojson', self.output_dir)


    def extract_chips(self):
        '''Extract chips from pan-sharpened image.'''

        # Make the pansharpened image directory the working directory
        os.chdir(self.ps_image_path)

        with open(join(self.ms_image_path, 'candidates.geojson')) as f:
            feature_collection = geojson.load(f)['features']

        if not os.path.exists('/chips/'):
            os.makedirs('/chips')

        for feat in feature_collection:
            # get bounding box of input polygon
            geom = feat['geometry']['coordinates'][0]
            f_id = feat['properties']['idx']
            xs, ys = [i[0] for i in geom], [i[1] for i in geom]
            ulx, lrx, uly, lry = min(xs), max(xs), max(ys), min(ys)

            # format gdal_translate command
            out_loc = join('/chips', str(f_id) + '.tif')

            cmd = 'gdal_translate -eco -q -projwin {0} {1} {2} {3} {4} {5} --config GDAL_TIFF_INTERNAL_MASK YES -co COMPRESS=JPEG -co PHOTOMETRIC=YCBCR -co TILED=YES'.format(str(ulx), str(uly), str(lrx), str(lry), self.ps_image, out_loc)

            try:
                subprocess.call(cmd, shell=True)
            except:
                # Don't throw error if chip is ouside raster
                print 'gdal_translate failed for the following command: ' + cmd # debug

        return True

    def get_ref_geojson(self):
        '''
        Create a reference geojson which only contains candidates for which chipping was successful.
        (Failures in chipping occur due to misaligment of the multispectral with the pansharpened.)
        '''

        with open(join(self.ms_image_path, 'candidates.geojson')) as f:
            data = geojson.load(f)

        # Get list of idxs in output directory
        chips = [f[:-4] for f in os.listdir('/chips/') if f.endswith('.tif')]
        feature_collection = data['features']
        valid_feats = []

        # Check if vector exists in chip directory
        for feat in feature_collection:
            if str(feat['properties']['idx']) in chips:
                valid_feats.append(feat)

        data['features'] = valid_feats
        output_file = '/chips/ref.geojson'

        with open(output_file, 'wb') as f:
            geojson.dump(data, f)


    def th_to_tf(self, X):
        '''
        Helper function to transform a normalized (3,h,w) image (theano ordering) to a
        (h,w,3) rgb image (tensor flow). This function is called int the prep chips
        function.
        '''
        rgb_array = np.zeros((X.shape[1], X.shape[2], 3), 'float32')
        rgb_array[...,0] = X[0]
        rgb_array[...,1] = X[1]
        rgb_array[...,2] = X[2]
        return rgb_array


    def prep_chips(self):
        '''
        Reshape chips to input size.
        Largest side of the chip is resized to 150px (input shape for model).
        The smaller side is zero-padded to create a square array.
        '''

        chips = [os.path.join('/chips/', chip) for chip in os.listdir('/chips/') if chip.endswith('.tif')]

        for chip in chips:

            # Format each chip
            raster_array = []

            if type(chip) == str:
                # Open chip
                img = gdal.Open(chip)

                # Get size info
                bands = img.RasterCount
                x, y = img.RasterXSize, img.RasterYSize
            else:
                bands = np.shape(chip)[0]
                x, y = np.shape(chip)[-1], np.shape(chip)[-2]

            resize_small = (150 * min(x,y)) / max(x,y)
            pad_size = 150 - resize_small

            if x >= y:
                resize = (resize_small, 150)
                p = ((0,pad_size),(0,0))
            else:
                resize = (150, resize_small)
                p = ((0,0),(0, pad_size))

            # Read chip as numpy array, reshape and pad
            for band in xrange(1, bands + 1):
                if type(chip) == str:
                    arr = img.GetRasterBand(band).ReadAsArray()
                else:
                    arr = chip[band - 1]
                arr = np.pad(imresize(arr, resize), p, 'constant', constant_values=0)
                raster_array.append(arr)

            # Save updated chip in npy format, remove old chip
            np.save(chip[:-4] + '.npy', self.th_to_tf(np.array(raster_array)))
            os.remove(chip)

    def generate_dataset(self):
        '''Generate chips in batches of 2000 for deploying.'''
        chips = [os.path.join('/chips',chip) for chip in os.listdir('/chips/') if chip.endswith('.npy')]

        for batch_ix in range(0, len(chips), 2000):
            # Generate batch
            batch = chips[batch_ix:batch_ix + 2000]
            X, fid = [], []

            for chip in batch:
                X.append(np.load(chip) / 255.)
                fid.append(os.path.split(chip)[-1][:-4])

            X = np.array(X)
            yield(X, fid)


    def deploy_model(self):
        '''Deploy model.'''
        feat_ids, preds = [],[]
        model = load_model('/model.h5')
        target_data_gen = self.generate_dataset()

        while True:
            try:
                x, fids = target_data_gen.next()
            except (StopIteration):
                break
            feat_ids += list(fids)
            preds += list(model.predict(x))

        # Format preds to put in geojson
        results = {}
        for i in range(len(feat_ids)):
            if preds[i][1] > self.threshold:
                results[feat_ids[i]] = [1, preds[i][1]]

        # Save results to geojson
        with open('/chips/ref.geojson') as f:
            data = geojson.load(f)

        boat_feats = []

        for feat in data['features']:
            try:
                res = results[str(feat['properties']['idx'])]
                feat['properties']['tank_certainty'] = np.round(res[1], 10).astype(float)
                boat_feats.append(feat)
            except (KeyError):
                next

        data['features'] = boat_feats

        with open(os.path.join(self.output_dir, 'results.geojson'), 'wb') as f:
            geojson.dump(data, f)


    def invoke(self):

        # Run protogen to get candidate bounding boxes
        print 'Detecting candidates...'
        candidates = self.extract_candidates()

        # Format vector file and chip from pan image
        print 'Chipping...'
        self.extract_chips()
        self.get_ref_geojson()

        # Format and pad chips
        self.prep_chips()

        # Deploy model
        print 'Deploying model...'
        self.deploy_model()


if __name__ == '__main__':
    with BoatDetector() as task:
        task.invoke()
