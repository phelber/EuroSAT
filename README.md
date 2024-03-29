# EuroSAT: Land Use and Land Cover Classification with Sentinel-2

![EuroSAT overview image](https://github.com/phelber/EuroSAT/blob/master/eurosat_overview_small.jpg?raw=true)

## Short Description

In this study, we address the challenge of land use and land cover classification using Sentinel-2 satellite images. The Sentinel-2 satellite images are openly and freely accessible provided in the Earth observation program Copernicus. We present a novel dataset based on Sentinel-2 satellite images covering 13 spectral bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images. We provide benchmarks for this novel dataset with its spectral bands using state-of-the-art deep Convolutional Neural Network (CNNs). With the proposed novel dataset, we achieved an overall classification accuracy of 98.57\%. The resulting classification system opens a gate towards a number of Earth observation applications. We demonstrate how this classification system can be used for detecting land use and land cover changes and how it can assist in improving geographical maps. The geo-referenced dataset EuroSAT is made publicly available [here](#).

### Dataset
The dataset is available via [Zenodo](https://zenodo.org/record/7711810#.ZAm3k-zMKEA).

##### Deprecated Hosting
* [EuroSAT Dataset (RGB)](https://madm.dfki.de/files/sentinel/EuroSAT.zip)
* [EuroSAT Dataset (MS)](https://madm.dfki.de/files/sentinel/EuroSATallBands.zip)

### Paper
* The full-text PDF is available via [ResearchGate](https://www.researchgate.net/publication/319463676_EuroSAT_A_Novel_Dataset_and_Deep_Learning_Benchmark_for_Land_Use_and_Land_Cover_Classification).

### FAQ

#### How to create the RGB version of an image using the multi-spectral version of an image using the GDAL command line tools?

One option to do this is:

```
gdal_translate --config GDAL_PAM_ENABLED NO -of JPEG -co QUALITY=100 -ot Byte -a_nodata 0 -scale 0 2750 1 255 -b 4 -b 3 -b 2 -of JPEG <input> <output>

```

You can also perform this process using your favorite geo lib (e.g. rasterio). Please note the scaling parameters set in the command line above, which lead to clipping and scaling effects.


### References

If you have used the EuroSAT dataset, please cite the following papers: 

[1] Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

```
@article{helber2019eurosat,
  title={Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2019},
  publisher={IEEE}
}
```

[2] Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. Patrick Helber, Benjamin Bischke, Andreas Dengel. 2018 IEEE International Geoscience and Remote Sensing Symposium, 2018.

```
@inproceedings{helber2018introducing,
  title={Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  booktitle={IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium},
  pages={204--207},
  year={2018},
  organization={IEEE}
}
```

### License
The dataset is licensed under the MIT license. In general, Sentinel data is free and open to the public under EU law. Please consider the [Copernicus Sentinel Data Terms and Conditions](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice) when using Copernicus Sentinel data.

