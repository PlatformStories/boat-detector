import numpy as np
import geojson, json
import time, os, shutil
import protogen
import utm
import cv2
import subprocess

from glob import glob
from osgeo import gdal
from scipy.misc import imresize
from keras.models import load_model
from gbdx_task_interface import GbdxTaskInterface
from os.path import join


def preprocess(data):
    '''
    Args: data: list of images
    Returns: data
          data: list of mean-adjusted & RGB -> BGR transformed images
    '''
    # BGR --> RGB
    data = data[:,:,:,[2,1,0]].astype('float32')

    data[:, :, :, 0] -= 103.939
    data[:, :, :, 1] -= 116.779
    data[:, :, :, 2] -= 123.68
    return data

def resize_image(path, side_dim):
    '''
    Args: path [string], rows, cols [int]
        path: path of an image; side_dim - columns and rows
        of resized image
    Returns: resized [numpy array rows x columns x 3]
        resized image
	(if the image is invalid, it returns a rowsxcols array of zeros)
    '''
    img = cv2.imread(path)
    try:
        x, y, _ = img.shape
        resize_small = int((side_dim * min(x,y)) / max(x,y))
        pad_size = side_dim - resize_small

        # Get resize and pad dimensions for chip
        if x >= y:
            resize = (resize_small, 224)
            p = ((0,0),(0,pad_size),(0,0))
        else:
            resize = (224, resize_small)
            p = ((0, pad_size), (0,0), (0,0))

        resized = np.pad(cv2.resize(img, resize), p, 'constant', constant_values=0)

    except:
        print 'Resizing can not be performed. Corrupt chip?'
        resized = np.zeros([rows, cols, 3], dtype=int)
    return resized


def get_utm_info(image):
    'Return UTM info of image. Image must be in UTM projection.'
    sample = gdal.Open(image)
    projection_info = sample.GetProjectionRef()
    where = projection_info.find('UTM zone') + 9
    utm_info = projection_info[where:where+3]
    utm_number, utm_letter = int(utm_info[0:2]), utm_info[2]
    return utm_number, utm_letter



class BoatDetector(GbdxTaskInterface):
    'Deploys a trained CNN classifier on protogen-generated candidate regions to determine which ones contain boats.'

    def __init__(self):

        GbdxTaskInterface.__init__(self)

        # Image inputs
        self.ms_dir = self.get_input_data_port('ms_image')
        self.ps_dir = self.get_input_data_port('ps_image')
        self.mask_dir = self.get_input_data_port('mask')

        # Point to imgs. If there are multiple tif's in multiple subdirectories, pick one.
        self.ms_image_path, self.ms_image = [(dp, f) for dp, dn, fn in os.walk(self.ms_dir) for f in fn if 'tif' in f][0]
        self.ps_image_path, self.ps_image = [(dp, f) for dp, dn, fn in os.walk(self.ps_dir) for f in fn if 'tif' in f][0]

        # Point to mask file if it's there. If there are multiple tif's in multiple subdirectories, pick one.
        try:
            self.mask_path, self.mask = [(dp, f) for dp, dn, fn in os.walk(self.mask_dir) for f in fn if 'tif' in f][0]
        except:
            self.mask_path, self.mask = None, None

        # String inputs
        self.threshold = float(self.get_input_string_port('threshold', '0.657'))
        self.with_mask = self.get_input_string_port('with_mask', 'false')
        self.dilation = int(self.get_input_string_port('dilation', '100'))
        self.min_linearity = float(self.get_input_string_port('min_linearity', '2.0'))
        self.max_linearity = float(self.get_input_string_port('max_linearity', '8.0'))
        self.min_size = int(self.get_input_string_port('min_size', '500'))
        self.max_size = int(self.get_input_string_port('max_size', '6000'))

        if self.with_mask in ['True', 'true', 't']:
            self.with_mask = True
        else:
            self.with_mask = False

        # Create output directories
        self.detections_dir = self.get_output_data_port('detections')
        if not os.path.exists(self.detections_dir):
            os.makedirs(self.detections_dir)
        self.candidates_dir = self.get_output_data_port('candidates')
        if not os.path.exists(self.candidates_dir):
            os.makedirs(self.candidates_dir)
        self.output_mask_dir = self.get_output_data_port('mask')
        if not os.path.exists(self.output_mask_dir):
            os.makedirs(self.output_mask_dir)

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

            # Remove small bodies in mask (could be land water bodies or shadows) with union find size filtering
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

            # Copy mask to output folder
            shutil.copy(uff.output, join(self.output_mask_dir, 'mask.tif'))

        # If mask is provided and with_mask is true then use the provided one
        elif self.mask and self.with_mask:
            make_mask = False
            shutil.copy(join(self.mask_path, self.mask), '.')

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
            if self.mask:
                msd.image = self.mask
    	    else:
                msd.image = uff.output
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

        # Rename geojson and copy it to candidates folder
        shutil.move(vbb.output, 'candidates.geojson')
        shutil.copy('candidates.geojson', self.candidates_dir)

        # Return to home dir
        os.chdir('/')


    def extract_chips(self):
        'Extract chips from pan-sharpened image.'

        # Get UTM info for conversion
        utm_num, utm_let = get_utm_info(join(self.ps_image_path, self.ps_image))

        with open(join(self.ms_image_path, 'candidates.geojson')) as f:
            feature_collection = geojson.load(f)['features']

        # Create directory for storing chips
        chip_dir = join(self.ps_image_path, '/chips/')
        if not os.path.exists(chip_dir):
            os.makedirs(chip_dir)

        for feat in feature_collection:
            # get bounding box of input polygon
            polygon = feat['geometry']['coordinates'][0]
            f_id = feat['properties']['idx']
            xs, ys = zip(*polygon)
            ulx, lrx, uly, lry = min(xs), max(xs), max(ys), min(ys)

            # Convert corner coords to UTM
            ulx, uly, utm_num1, utm_let1 = utm.from_latlon(uly, ulx, force_zone_number=utm_num)
            lrx, lry, utm_num2, utm_let2 = utm.from_latlon(lry, lrx, force_zone_number=utm_num)

            # format gdal_translate command
            out_loc = join(chip_dir, str(f_id) + '.tif')

            cmd = 'gdal_translate -eco -q -projwin {0} {1} {2} {3} {4} {5} '\
                  '--config GDAL_TIFF_INTERNAL_MASK YES -co TILED='\
                  'YES'.format(str(ulx), str(uly), str(lrx), str(lry),
                               join(self.ps_image_path, self.ps_image), out_loc)

            try:
                subprocess.call(cmd, shell=True)
            except:
                # Don't throw error if chip is ouside raster
                print 'gdal_translate failed for the following command: ' + cmd # debug

        return True


    def deploy_model(self):
        'Deploy model.'
        model = load_model('/model.h5')
        boats = {}
        chips = glob(join('chips', '*.tif'))

        # Classify chips in batches
        indices = np.arange(0, len(chips), 100)
        no_batches = len(indices)

        for no, index in enumerate(indices):
            batch = chips[index: (index + 100)]
            X = preprocess(np.array([resize_image(chip, 224) for chip in batch]))
            fids = [os.path.split(chip)[-1][:-4] for chip in batch]

            # Deploy model on batch
            print 'Classifying batch {} of {}'.format(no+1, no_batches)
            t1 = time.time()
            yprob = list(model.predict_on_batch(X))

            # create dict of boat fids and certainties
            for ix, pred in enumerate(yprob):
                if pred[0] > self.threshold:
                    boats[fids[ix]] = pred[0]

            t2 = time.time()
            print 'Batch classification time: {}s'.format(t2-t1)

        # Save results to geojson
        with open(join(self.ms_image_path, 'candidates.geojson')) as f:
            data = geojson.load(f)

        # Save all boats to output geojson
        boat_feats = []
        for feat in data['features']:
            try:
                res = boats[str(feat['properties']['idx'])]
                feat['properties']['boat_certainty'] = np.round(res, 10).astype(float)
                boat_feats.append(feat)
            except (KeyError):
                continue

        data['features'] = boat_feats

        with open(join(self.detections_dir, 'detections.geojson'), 'wb') as f:
            geojson.dump(data, f)


    def invoke(self):

        # Run protogen to get candidate bounding boxes
        print 'Detecting candidates...'
        candidates = self.extract_candidates()

        # Format vector file and chip from pan image
        print 'Chipping...'
        self.extract_chips()

        # Deploy model
        print 'Deploying model...'
        self.deploy_model()


if __name__ == '__main__':
    with BoatDetector() as task:
        task.invoke()
