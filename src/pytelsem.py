""" 
Housing the port of TELSEM smissivity atlas into python
"""

import os
import numpy as np
import geopandas
import pandas as pd
from shapely.geometry import Point

# Error checking
errorstatus_fatal = -1

class Telsem2AtlasData:
    def __init__(self, path:str):
        # number of lines in the atlas
        self.ndat = None
        # number of channels in the atlas
        self.nchan = 7
        # name of the atlas (including version number)
        self.path = path

        # Directory where the atlas data is stored
        self.dir = os.path.dirname(self.path)
        self.name = os.path.basename(self.path)
        
        # resolution of the atlas (equal-area)
        self.dlat = 0.25

        # ALLOCATE VARIABLES
        self.emis = np.zeros((self.ndat, self.nchan), dtype=np.float64)
        self.emis_err = np.zeros((self.ndat, self.nchan), dtype=np.float64)
        self.class1 = np.zeros(self.ndat, dtype=np.int32)
        self.class2 = np.zeros(self.ndat, dtype=np.int32)
        self.cellnum = np.zeros(self.ndat, dtype=np.int32)
        self.correspondance = np.full(660066, -777, dtype=np.int32)

        # number of cells per lat band
        self.ncells = None
        # the first cellnumber of lat band
        self.firstcell = None
        # limits of the spatial domain (flagged if global)
        self.lat1 = None
        self.lat2 = None
        self.lon1 = None
        self.lon2 = None
        # Emissivities
        self.emis = None  # emis(ndat,nchan)
        # Correlations
        self.correl = None  # correl(10,nchan,nchan)
        # Emissivity uncertainties (std)
        self.emis_err = None  # emis_err(ndat,nchan)
        # Surface class
        self.class1 = None
        self.class2 = None
        # cellnumber of each of the pixels in the atlas
        self.cellnum = None
        # "Correspondance" vector indicating that for the ith element, the j so that EMIS(j,...) is the emissivity of cellnumber i.
        self.correspondance = None  # correspondance(660066)
        errorstatus_fatal = -1

        # Added:
        self.coordinates = None

    def __str__(self):
        message = f"Telsem Atlas at \n {self.path} \n"
        message += f"Atlas name: {os.path.basename(self.path)} \n"
        if self.coordinates:
            message += "Coordinates loaded \n"

        return message

    def equare(self):
        """ 
        Equal area computations
        """
        dlat = self.dlat
        ncells = self.ncells
        firstcell = self.firstcell

        maxlat = int(180.0 / dlat)
        maxlon = int(360.0 / dlat)

        tocell = np.zeros((maxlon, maxlat), dtype=np.int32)

        REARTH = 6371.2
        PI = np.pi
        RCELAT = (dlat * PI) / 180.0
        TOTCEL = 0

        HEZON = REARTH * np.sin(RCELAT)
        AEZON = 2.0 * PI * REARTH * HEZON
        AECELL = (AEZON * dlat) / 360.0

        MAXLT2 = maxlat // 2
        for lat in range(1, MAXLT2 + 1):
            XLATB = (lat - 1) * dlat
            XLATE = XLATB + dlat
            RLATB = (2.0 * PI * XLATB) / 360.0
            RLATE = (2.0 * PI * XLATE) / 360.0

            HTB = REARTH * np.sin(RLATB)
            HTE = REARTH * np.sin(RLATE)
            HTZONE = HTE - HTB
            AZONE = 2.0 * PI * REARTH * HTZONE

            RCELLS = AZONE / AECELL
            ICELLR = int(RCELLS + 0.5)

            TOTCEL += 2 * ICELLR

            lat1 = lat + MAXLT2
            lat2 = MAXLT2 + 1 - lat
            ncells[lat1 - 1] = ICELLR
            ncells[lat2 - 1] = ICELLR

        numcel = 0
        for lat in range(1, maxlat + 1):
            numcls = ncells[lat - 1]
            for lon in range(1, numcls + 1):
                numcel += 1
                tocell[lon - 1, lat - 1] = numcel

        for i in range(1, maxlat + 1):
            if i == 1:
                firstcell[i - 1] = 1
            else:
                firstcell[i - 1] = firstcell[i - 2] + ncells[i - 2]


    def rttov_readmw_atlas(
        self,
        verbose=False,
        lat1=-777,
        lat2=-777,
        lon1=-777,
        lon2=-777,
    ):


        # TRANSITORY VARIABLES
        iiin = 21  # unit for input
        cellnum = 0
        ssmi = np.zeros(2 * 7, dtype=np.float64)
        lat, lon = 0.0, 0.0
        cur_class1, cur_class2 = 0, 0
        take = 0  # flag to take or not the pixel atlas for location constraints
        msg = ""

        # initialisation
        err = 0

        if self.emis is not None:
            err = errorstatus_fatal
            print("TELSEM2 atlas data structure already allocated")
            return err

        self.lat1 = lat1
        self.lat2 = lat2
        self.lon1 = lon1 % 360.0
        self.lon2 = lon2 % 360.0

        # ALLOCATION SPECIFIC TO SSMI ATLAS
        self.nchan = 7
        self.dlat = 0.25
        self.ncells = np.zeros(int(180.0 / self.dlat), dtype=np.int32)
        self.firstcell = np.zeros(int(180.0 / self.dlat), dtype=np.int32)

        # Call the method to produce all the equal area calculations
        self.equare()

        # DEFINING NUMBER OF DATA
        if verbose:
            print("Reading number of data in atlas...")

        try:
            iiin_file = open(
                self.path, "r"
            )
            j = int(iiin_file.readline().strip())
            self.ndat = j
            if verbose:
                print(f"Nb data={self.ndat}")
        except IOError as e:
            print(f"Error opening the monthly input file: {e}")
            err = errorstatus_fatal
            return err

        ipos = 0
        for line in iiin_file:
            parts = line.strip().split()
            cellnum, ssmi_values, cur_class1, cur_class2 = (
                int(parts[0]),
                list(map(float, parts[1:15])),
                int(parts[15]),
                int(parts[16]),
            )

            take = 1
            if lat1 is not -777:
                if not (lat1 <= lat <= lat2 and lon1 <= lon <= lon2):
                    take = 0

            if cur_class1 > 0 and cur_class2 > 0 and ipos < self.ndat and take == 1:
                ipos += 1
                self.emis[ipos - 1, :] = ssmi_values[:7]
                self.emis_err[ipos - 1, :] = np.sqrt(ssmi_values[7:])
                self.cellnum[ipos - 1] = cellnum
                self.class1[ipos - 1] = cur_class1
                self.class2[ipos - 1] = cur_class2
                self.correspondance[cellnum - 1] = ipos

        self.ndat = ipos
        iiin_file.close()

        # Correlation of uncertainties
        self.correl = np.zeros((10, self.nchan, self.nchan), dtype=np.float64)

        if verbose:
            print("Reading classes...")

        try:
            iiin_file = open(os.path.join(self.dir, "correlations"), "r")
            for i in range(10):
                iiin_file.readline()  # skip lines
                for j in range(7):
                    self.correl[i, j, :] = np.array(
                        list(map(float, iiin_file.readline().strip().split()))
                    )
        except IOError as e:
            print(f"Error opening the correlations input file: {e}")
            err = errorstatus_fatal
            return err
        finally:
            iiin_file.close()
        return err
    
    def get_all_coordinates(self):
        """ 
        Populate the coordinates field, giving pais of lat-lon points in a list.
        1 to 1 correspondence to self.cellnum
        """
        self.coordinates = [self.get_coordinates(c) for c in self.cellnum]
        pass

    def get_coordinates(self, cellnum: int) ->tuple[float, float]:
        """ 
        Create an equivalent geodataframe from the atlas data
        Columns: 
            Geometry (points with the lon-lat coordinates)
            7 emmisivity bands
            7 error estimation, one for each of the bands.
            2 surface classification flags for each point.
        """

        res_lat = self.dlat

        index_lat_max = int(180 / res_lat) - 1

        if cellnum >= self.firstcell[index_lat_max]:
            index_lat = index_lat_max
            lat = (index_lat - 0.5) * res_lat - 90
            index_lon = cellnum - self.firstcell[index_lat_max] + 1
            lon = (index_lon - 0.5) * (360.0 / self.ncells[index_lat])

        else:
            for i in range(1, index_lat_max + 1):
                if cellnum >= self.firstcell[i] and cellnum < self.firstcell[i + 1]:
                    index_lat = i
                    lat = (index_lat - 0.5) * res_lat - 90
                    index_lon = cellnum - self.firstcell[i] + 1
                    lon = (index_lon - 0.5) * (360.0 / self.ncells[index_lat])

        return lat, lon
    
    def to_geopandas(self) -> geopandas.GeoDataFrame:
        """ 
        Export the atlas data
        """
        if self.coordinates == None:
            print("Calculating coordinates as points")
            self.get_all_coordinates()

        lats = [c[0] for c in self.coordinates]
        lons = [(c[1] - 180) % 360 for c in self.coordinates]
        geometry = [Point(lon,lat) for lat,lon in zip(lats,lons)]

        gdf = geopandas.GeoDataFrame(geometry=geometry)

        # Create dataframes for emisivities and errors:
        emisdf = pd.DataFrame(
            self.emis,
            columns=[
                "Emis19V",
                "Emis19H",
                "Emis22V",
                "Emis37V",
                "Emis37H",
                "Emis85V",
                "Emis85H",
            ],
        )

        errdf = pd.DataFrame(
            self.emis_err,
            columns=[
                "VarEmis19V",
                "VarEmis19H",
                "VarEmis22V",
                "VarEmis37V",
                "VarEmis37H",
                "VarEmis85V",
                "VarEmis85H",
            ],
        )
        # surface class information
        sc1df = pd.DataFrame(self.class1, columns=["Surface_class1"])
        sc2df = pd.DataFrame(self.class2, columns=["Surface_class2"])

        return pd.concat([gdf, emisdf, errdf, sc1df, sc2df], axis=1)
    
    # TODO: implement/port the spacial interpolation algorithm.
    def spacial_interpolation(self,lon:float, lat:float, variable:str=None) -> tuple[float, float]:
        """ 
        Using data from the atlas, interpolate for an arbitrary coordinate
        param variable: if it exists in the atlas, calculate the value only
            for that emissivity value and its error

        return: emissivity value, propagatet error
        """
        emiss, emiss_err = None, None

        

        return emiss, emiss_err