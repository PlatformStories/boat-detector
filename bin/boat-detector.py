import numpy as np
import geojson, json
import time, os, shutil
import protogen
import utm
import cv2
import subprocess

from glob import glob
from osgeo import gdal, osr
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
    'Return UTM number and proj4 format of utm projection. Image must be in UTM projection.'
    sample = gdal.Open(image)
    prj = sample.GetProjectionRef()
    srs = osr.SpatialReference(wkt=prj)
    return srs.GetUTMZone(), srs.ExportToProj4()

def execute_this(command):
    proc = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.communicate()


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
        self.threshold = float(self.get_input_string_port('threshold', '0.5'))
        self.erosion = int(self.get_input_string_port('erosion', '100'))
        self.min_linearity = float(self.get_input_string_port('min_linearity', '2.0'))
        self.max_linearity = float(self.get_input_string_port('max_linearity', '8.0'))
        self.min_size = int(self.get_input_string_port('min_size', '500'))
        self.max_size = int(self.get_input_string_port('max_size', '6000'))

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

        # If mask is not provided then make one
        if not self.mask:

            print 'Creating water mask'
            make_mask = True

            print '- compute extent in utm coordinates'
            img = gdal.Open(self.ms_image)
            ulx, xres, xskew, uly, yskew, yres  = img.GetGeoTransform()
            lrx = ulx + (img.RasterXSize * xres)
            lry = uly + (img.RasterYSize * yres)
            utm_number, utm_proj4 = get_utm_info(self.ms_image)
            print '- UTM {}: ulx, uly, lrx, lry: {} {} {} {}'.format(utm_number, ulx, uly, lrx, lry)
            northern = (utm_number>0)

            # Get extent in EPSG:4326
            y1, x1 = utm.to_latlon(ulx, uly, zone_number=abs(utm_number), northern=northern)
            y2, x2 = utm.to_latlon(lrx, uly, zone_number=abs(utm_number), northern=northern)
            y3, x3 = utm.to_latlon(lrx, lry, zone_number=abs(utm_number), northern=northern)
            y4, x4 = utm.to_latlon(ulx, lry, zone_number=abs(utm_number), northern=northern)

            print '- clip water polygons to raster extent and reproject clipped shapefile to UTM'
            buffer_min, buffer_max = 0.99, 1.01
            command = 'ogr2ogr {} {} -spat {} {} {} {} -clipsrc spat_extent'.format('/water-polygons/water.shp',
                                                                                    '/water-polygons/water_polygons.shp',
                                                                                    buffer_min*min(x1, x4),
                                                                                    buffer_min*min(y3, y4),
                                                                                    buffer_max*max(x2, x3),
                                                                                    buffer_max*max(y1, y2))
            out, err = execute_this(command)
            command = """ogr2ogr {} {} -s_srs EPSG:4326 -t_srs '{}'""".format('/water-polygons/water-utm.shp',
                                                                              '/water-polygons/water.shp',
                                                                              utm_proj4)
            out, err = execute_this(command)

            print '- burn mask'
            command = 'gdal_rasterize -ot Byte -burn 255 -te {} {} {} {} -tr {} {} {} {}'.format(ulx, lry, lrx, uly, xres, yres, '/water-polygons/water-utm.shp', 'mask.tif')
            out, err = execute_this(command)

            # Copy mask to output folder
            shutil.copy('mask.tif', self.output_mask_dir)

        # If mask is provided then use the provided one
        elif self.mask:
            make_mask = False
            shutil.copy(join(self.mask_path, self.mask), 'mask.tif')

        # Compute band dissimilarity map with radex
        print 'Compute dissimilarity map'
        rbd = protogen.Interface('radex_scalar', 'band_dissimilarity')
        rbd.radex_scalar.band_dissimilarity.type = 'max'
        rbd.radex_scalar.band_dissimilarity.threshold = 1
        rbd.image = self.ms_image
        rbd.image_config.bands = range(1, no_bands+1)
        rbd.execute()

        # Apply erosion to water mask
        print 'Eroding water mask'
        msd = protogen.Interface('morphology', 'structural')
        msd.morphology.structural.operator = 'erosion'
        msd.morphology.structural.structuring_element = 'disk'
        msd.morphology.structural.radius1 = self.erosion
        msd.image = 'mask.tif'
        msd.image_config.bands = [1]
        msd.execute()

        # Apply the mask on the dissimilarity map
        print 'Apply mask on dissimilarity map'
        im = protogen.Interface('image_processor', 'masking')
        im.image_processor.masking.method = 'inclusion'
        im.image_processor.masking.tree_type = 'raw'
        im.image = rbd.output
        im.image_config.bands = [1]
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
        mf.image = im.output
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
        utm_num, utm_proj4 = get_utm_info(join(self.ps_image_path, self.ps_image))

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

            command = 'gdal_translate -eco -q -projwin {} {} {} {} {} {} '\
                      '--config GDAL_TIFF_INTERNAL_MASK YES -co TILED='\
                      'YES'.format(ulx, uly, lrx, lry,
                                   join(self.ps_image_path, self.ps_image), out_loc)

            try:
                execute_this(command)
            except:
                # Don't throw error if chip is ouside raster
                print 'gdal_translate failed for the following command: ' + command # debug

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
