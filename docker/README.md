Each step starts from the `yolact_edge` folder:
- Build:

`cd docker`

`./build.sh`
- Start work:
``./docker/start.sh `pwd` <path to dataset on your computer>`` (dataset will be available inside the container in a /datasets folder)


Connect to runnig container (if you need more than one terminal to the container):
`./docker/into.sh`

Container uses `yolact_edge` folder as docker volume, so all your modifications will be save after closing session.
