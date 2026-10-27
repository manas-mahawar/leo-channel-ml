# References and data attribution

## Paper replicated

S. A. El-Sayed, M. Ibrahim, H. El-Hennawy and E. S. Abbas, "Prediction of
channel propagation impairments in LEO satellite networks using machine
learning", *EURASIP Journal on Wireless Communications and Networking*,
2026:102 (received 15 January 2026, accepted 6 August 2026).
doi:[10.1186/s13638-026-02667-w](https://doi.org/10.1186/s13638-026-02667-w).
Open access under CC BY-NC-ND 4.0. The paper's numbers are quoted in this
repository as reference values with attribution; its PDF, text and figures
are not redistributed here.

(Details taken from the article's first page and back matter.)

## Data

* **Open-Meteo Historical Weather API** (hourly ERA5-based weather, the
  source of all model inputs). P. Zippenfenig (2023), *Open-Meteo.com Weather
  API*, Zenodo, doi:[10.5281/zenodo.7970649](https://doi.org/10.5281/zenodo.7970649).
  Data licence: CC BY 4.0, which requires attribution ("Weather data by
  Open-Meteo.com", <https://open-meteo.com/>). Source:
  <https://open-meteo.com/en/licence>.
* **ERA5 reanalysis** (underlying the Open-Meteo data; ERA5 total column cloud
  liquid water from the Copernicus Climate Data Store was used only to fit
  the cloud liquid-water proxy). H. Hersbach et al. (2020), "The ERA5 global
  reanalysis", *Quarterly Journal of the Royal Meteorological Society*
  146(730), 1999-2049, doi:[10.1002/qj.3803](https://doi.org/10.1002/qj.3803).
  ERA5 is produced by ECMWF for the Copernicus Climate Change Service (C3S);
  CDS downloads are subject to the Copernicus licence accepted with a CDS
  account. The raw CDS files are not redistributed here.

## Propagation models

ITU-R Recommendations as implemented in `itur` 0.4.0 (versions as used by
that release):

* ITU-R P.676-12, *Attenuation by atmospheric gases and related effects*
* ITU-R P.840-8, *Attenuation due to clouds and fog*
* ITU-R P.838-3, *Specific attenuation model for rain for use in prediction methods*
* ITU-R P.618-13, *Propagation data and prediction methods required for the
  design of Earth-space telecommunication systems*
* ITU-R P.839-4, *Rain height model for prediction methods*
* ITU-R P.1511, *Topography for Earth-space propagation modelling* (station height)

The paper cites P.676-13 and P.618-14; `itur` 0.4.0 implements the versions
above (documented as a deviation in REPLICATION.md).

## Software

* **ITU-Rpy** (`itur` on PyPI), I. del Portillo, *ITU-Rpy: A python
  implementation of the ITU-R P. Recommendations to compute atmospheric
  attenuation in slant and horizontal paths*, MIT licence.
  Documentation: <https://itu-rpy.readthedocs.io/>.
* TensorFlow / Keras, NumPy, pandas, SciPy, scikit-learn, matplotlib,
  xarray (versions in `requirements.txt`).
