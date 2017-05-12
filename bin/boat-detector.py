# Inputs:
# MS image
# PANSH image

# Note: pan image must be in lat lon (EPSG:4326) for now. this matches the crs output by protogen.
# Will likely want to switch to UTM, but this is tricky given the different codes. can also
# use gdalwarp on chips, but this will cause the chipping section to take at least twice as long.


import numpy as np
import geojson
import subprocess, os
import protogen

from osgeo import gdal
from scipy.misc import imresize
from keras.models import load_model
from gbdx_task_interface import GbdxTaskInterface


class DetectShips(GbdxTaskInterface):
    '''
    Deploys a trained CNN on Protogen-generated target chips to deteremine which contain
        ships
    '''

    def __init__(self):
        '''
        Get inputs
        '''
        GbdxTaskInterface.__init__(self)

        self.ms_dir = self.get_input_data_port('ms_image')
        self.ms_image = os.path.join(self.ms_dir, [i for i in os.listdir(self.ms_dir) if i.endswith('.tif')][0])
        self.pan_dir = self.get_input_data_port('pan_image')
        self.pan_image = os.path.join(self.pan_dir, [i for i in os.listdir(self.pan_dir) if i.endswith('.tif')][0])
        self.outdir = self.get_output_data_port('results')
        os.makedirs(self.outdir)


    ### PROTOGEN SECTION ###
    def extract_ships(self):
        '''
        Use protogen to generate potential ship locations
        '''
        # Create masked MS image
        l = protogen.Interface('lulc','masks')
        l.lulc.masks.type = 'single'
        l.lulc.masks.switch_no_data = False
        l.lulc.masks.switch_water = False
        l.lulc.masks.switch_vegetation = True
        l.lulc.masks.switch_clouds = True
        l.lulc.masks.switch_bare_soil = True
        l.lulc.masks.switch_shadows = True
        l.lulc.masks.switch_unclassified = False
        l.image_config.bands = [1, 2, 3, 4, 5, 6, 7, 8]
        l.image = self.ms_image
        l.execute()

        # Remove LULC noise
        a = protogen.Interface('morphology', 'connected_area_filter')
        a.morphology.connected_area_filter.operator = 'opening'
        a.morphology.connected_area_filter.opening_size_threshold = 10000.0
        a.morphology.connected_area_filter.closing_size_threshold = 0.0
        a.image_config.bands = [1]
        a.image = os.path.split(l.output)[-1]
        a.execute()

        # Close clouds (?)
        c = protogen.Interface('morphology', 'structural')
        c.morphology.structural.operator = 'closing'
        c.morphology.structural.structuring_element = 'disk'
        c.morphology.structural.radius1 = 10
        c.image_config.bands = [1]
        c.image = os.path.split(a.output)[-1]
        c.execute()

        # Extract ships from original image + mask
        p = protogen.Interface('extract', 'vehicles')
        p.extract.vehicles.type = 'ships'
        p.extract.vehicles.visualization = 'binary'
        p.extract.vehicles.max_length = 1500.0
        p.extract.vehicles.max_width = 100.0
        p.extract.vehicles.min_length = 50.0
        p.extract.vehicles.min_width = 10.0
        p.extract.vehicles.threshold = 100.0
        p.image_config.bands = [1, 2, 3, 4, 5, 6, 7, 8]
        p.mask_config.bands = [1]
        p.image = self.ms_image
        p.mask = os.path.split(c.output)[-1]
        p.execute()

        # Get bbox vectors of ships
        v = protogen.Interface('vectorizer', 'bounding_box')
        v.image_config.bands = [1]
        v.vectorizer.bounding_box.filetype = 'geojson'
        v.vectorizer.bounding_box.target_bbox = False
        v.vectorizer.bounding_box.target_centroid = True
        v.vectorizer.bounding_box.processor = True
        v.athos.tree_type = 'union_find'
        v.athos.area.export = [1]
        v.image = os.path.split(p.output)[-1]
        v.execute()

        return v.output


    ### CHIPPING SECTION ###
    def chip_vectors(self, vector_file):
        '''
        Create and execute gdal_translate commands for extracting each chip from
            pansharpened image
        '''
        os.makedirs('/chips/')

        with open(vector_file) as f:
            feature_collection = geojson.load(f)['features']

        for feat in feature_collection:
            # get bounding box of input polygon
            geom = feat['geometry']['coordinates'][0]
            f_id = feat['properties']['idx']
            xs, ys = [i[0] for i in geom], [i[1] for i in geom]
            ulx, lrx, uly, lry = min(xs), max(xs), max(ys), min(ys)

            # format gdal_translate command
            out_loc = os.path.join('/chips', str(f_id) + '.tif')

            cmd = 'gdal_translate -eco -q -projwin {0} {1} {2} {3} {4} {5} --config GDAL_TIFF_INTERNAL_MASK YES -co COMPRESS=JPEG -co PHOTOMETRIC=YCBCR -co TILED=YES'.format(str(ulx), str(uly), str(lrx), str(lry), self.pan_image, out_loc)
            print cmd # debug

            try:
                subprocess.call(cmd, shell=True)
            except:
                # Don't throw error if chip is ouside raster
                print 'gdal_translate failed for the following command: ' + cmd # debug

        return True


    def get_ref_geojson(self, vector_file):
        '''
        create a reference geojson with only features in chips output directory.

        There's a chance not all vectors were chipped out of the pan image (the pan and ms
            images don't always overlap perfectly). The ref geojson will avoid errors
            while generating chips for deploying.
        '''
        with open(vector_file) as f:
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


    ### PREPARE CHIP DATA SECTION ###
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
        Open and reshape chips to input size. Largest side of the chip will be resized to
            150px (input shape for model), the smaller side will be zero-padded to create
            a square array.
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
        '''
        generate chips in batches of 2000 for deploying
        '''
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
        '''
        deploy the model
        '''
        feat_ids, preds = [],[]
        model = load_model('model.h5')
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
            if preds[i][0] > 0.5:
                results[feat_ids[i]] = [0, preds[i][1]]
            else:
                results[feat_ids[i]] = [1, preds[i][1]]

        # Save results to geojson
        with open('/chips/ref.geojson') as f:
            data = geojson.load(f)

        for feat in data['features']:
            res = results[str(feat['properties']['idx'])]
            feat['properties']['CNN_class'] = res[0]
            feat['properties']['tank_certainty'] = np.round(res[1], 10).astype(float)

        with open(os.path.join(self.outdir, 'results.geojson'), 'wb') as f:
            geojson.dump(data, f)


    def invoke(self):
        '''
        Execute task
        '''
        # Run protogen to get target chips
        ship_vectors = self.extract_ships()

        # Format vector file and chip from pan image
        self.chip_vectors(ship_vectors)
        self.get_ref_geojson(ship_vectors)

        # Format and pad chips
        self.prep_chips()

        # Deploy model
        self.deploy_model()


if __name__ == '__main__':
    with DetectShips() as task:
        task.invoke()
