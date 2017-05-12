# boat-detector

A GBDX task that detects boats. Boats include ships, vessels, and all types of watercraft.

The inputs to the task are a 4/8-band multispectral image and its pansharpened counterpart. The output is a geojson file with the boat bounding boxes. The task first detects elongated anomalies in the water and then deploys a trained Keras model to single out the anomalies which correspond to boats.

<img src='images/boat-detector.png' width=700>

## Run

This is a sample workflow to detect vessels in the New York area. The required input imagery is found in S3.

1. Within an iPython terminal create a GBDX interface an specify the task input location:  

    ```python
    from gbdxtools import Interface
    from os.path import join
    import uuid

    gbdx = Interface()

    input_location = 's3://gbd-customer-data/58600248-2927-4523-b44b-5fec3d278c09/platform-stories/boat-detector/'
    ```

2. Create a task instance and set the required [inputs](#inputs):  

    ```python
    ship_task = gbdx.Task('boat-detector')
    ship_task.inputs.pan_image = join(input_location, 'pan_image')
    ship_task.inputs.ms_image = join(input_location, 'ms_image')
    ```

3. Initialize a workflow and specify where to save the output:  

    ```python
    ship_wf = gbdx.Workflow([ship_task])
    random_str = str(uuid.uuid4())
    output_location = join('platform-stories/trial-runs', random_str)

    ship_wf.savedata(ship_task.outputs.results, join(output_location, 'ship_detections'))
    ```

5. Execute the workflow:  

    ```python
    ship_wf.execute()
    ```

6. Track the status of the workflow as follows:

    ```python
    ship_wf.status
    ```


## Inputs

GBDX input ports can only be of "Directory" or "String" type. Booleans, integers and floats are passed to the task as strings, e.g., "True", "10", "0.001".

| Name  | Type | Description | Required |
|---|---|---|---|
| ms_image | directory | Contains a 4/8-band multispectral GeoTiff. This directory should only contain one image, otherwise a file will be selected arbitrarily. | True |
| pan_image | directory | Contains the pansharpened counterpart of the multispectal image. This directory should only contain one image, otherwise a file will be selected arbitrarily. | True |

## Outputs

| Name  | Type | Description |
|---|---|---|
| results | directory | Contains a geojson with the boat bounding boxes. |


## Development

### Build the Docker Image

You need to install [Docker](https://docs.docker.com/engine/installation).

Clone the repository:

```bash
git clone https://github.com/platformstories/boat-detector
```

Then build the image locally. Building requires input environment variables for protogen and GBDX AWS credentials. You will need to contact kostas.stamatiou@digitalglobe.com for access to Protogen.

```bash
cd boat-detector
docker build --build-arg PROTOUSER=<GitHub username> \
    --build-arg PROTOPASSWORD=<GitHub pawwsord> \
    --build-arg AWS_ACCESS_KEY_ID=<AWS access key> \
    --build-arg AWS_SECRET_ACCESS_KEY=<AWS secret key> \
    --build-arg AWS_SESSION_TOKEN=<AWS session token> \
    -t boat-detector .
```

### Try out locally

Create a container in interactive mode and mount the sample input under `/mnt/work/input/`:

```bash
docker run -v full/path/to/sample-input:/mnt/work/input -it boat-detector
```

Then, within the container:

```bash
python /boat-detector.py
```

### Docker Hub

Login to Docker Hub:

```bash
docker login
```

Tag your image using your username and push it to DockerHub:

```bash
docker tag boat-detector yourusername/boat-detector
docker push yourusername/boat-detector
```

The image name should be the same as the image name under containerDescriptors in boat-detector.json.

Alternatively, you can link this repository to a [Docker automated build](https://docs.docker.com/docker-hub/builds/). Every time you push a change to the repository, the Docker image gets automatically updated.

### Register on GBDX

In a Python terminal:
```python
from gbdxtools import Interface
gbdx=Interface()
gbdx.task_registry.register(json_filename="boat-detector.json")
```

Note: If you change the task image, you need to reregister the task with a higher version number in order for the new image to take effect. Keep this in mind especially if you use Docker automated build.
