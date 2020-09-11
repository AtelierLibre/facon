'''Facón uses KMeans cluster centroids to subdivide polygonal geometry in a GeoDataFrame

The main function is:

subdivide_zones(gdf, area_m2=500000)

Which splits polygons held in a GeoDataFrame towards the
target area.

version 0.0.3 - 11 Sep 2020
'''

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import asMultiPoint, LineString, MultiPolygon, Polygon, box
from shapely.ops import split, unary_union

from sklearn.cluster import KMeans

from libpysal.cg.voronoi  import voronoi, voronoi_frames

__version__ = '0.0.2'


def _point_fill_poly(polygon, target_area_m2, points_per_cell, seed):
    """
    Returns a shapely MultiPoint object that fills the original polygon.
    
    Fills the polygon's bounding box with randomly placed points then
    clips this to the polygon's outline.
    
    If the goal is to create walkable zones approximately 1km2 place
    points approximately every 100x100m.
    
    Assumes the polygon is in metres.
    
    Parameters
    ----------
    polygon: Polygon
        the polygon to fill with points
    target_area_m2: int
        Target area for subdivisions in square meters
    points_per_cell : int, optional
        the approximate number of points that will be in each cluster
    seed : int, optional
        seed for reproducibility
    
    Returns
    -------
    clipped_multi_point : MultiPoint
        a shapely MultiPoint object clipped to the input polygon
    """
    if seed is None:
        seed = 42
    np.random.seed(seed)

    if points_per_cell is None:
        points_per_cell = 100

    # calculate the number of random points to generate
    # calcualtes the area of the bouding box
    minx, miny, maxx, maxy = polygon.bounds
    bounding_box_area = (maxx-minx)*(maxy-miny)

    # calculates how many of the target_area cells would fill the bounding box
    num_cells = bounding_box_area/target_area_m2

    # multiplies that number by the number of points per cell to determine
    # the total number of points to generate
    num_points = int(num_cells*points_per_cell)

    # create x coordinates
    x = np.random.uniform(minx,maxx,num_points)
    # create y coordinates
    y = np.random.uniform(miny,maxy,num_points)

    # combine the arrays into a 2D coordinate array
    xy = np.array((x,y)).T

    # convert the coordinate array into a shapely MultiPoint object
    multipoint_ = asMultiPoint(xy)

    # clip the multipoint to the original input polygon
    clipped_multipoint = multipoint_.intersection(polygon)

    return clipped_multipoint


def _create_cluster_centroids(polygon_, clipped_multipoint, target_area_m2, random_state=0):
    '''
    Generate clusters from points in a polygon and return the cluster centroids

    Takes a polygon and a multipoint object that fills it and clusters the points
    into a number of regions defined by dividing the polygon's area by a target area.
    Returns the centroids of the clusters.

    Parameters
    ----------
    polygon_ : Polygon
        the polygon to subdivide
    clipped_multipoint : MultiPoint
        the MultiPoint object that fills the polygon
    target_area_m2: int
        Target area for subdivisions in square meters
    random_state : int
        set the random state for reproducibility

    Returns
    -------
    cluster_centroids : MultiPoint
        single shapely MultiPoint containing the cluster centroids
    '''

    # divide the area of the polygon by the target area to
    # determine the number of clusters (rounds to nearest whole number)
    n_clusters = round(polygon_.area / target_area_m2)

    # prevent calls with zero clusters
    if n_clusters < 1:
        n_clusters = 1

    # extract multipoint coordinates as array to cluster
    X = np.array(clipped_multipoint)

    # instantiate and fit KMeans to the array
    kmeans = KMeans(n_clusters, random_state=random_state).fit(X)

    # extract the cluster_centers_ and combine them into a MultiPoint geometry
    cluster_centroids = asMultiPoint(kmeans.cluster_centers_)

    return cluster_centroids


def _perpendicular_bisector(multipoint):
    '''
    Creates a line that is the perpendicular bisector of a multipoint
    containing two points.
    
    Parameters
    ----------
    multipoint: MultiPoint
        shapely MultiPoint containing two points

    Returns
    -------
    _perpendicular_bisector : LineString
        shapely LineString bisecting the two points
    '''

    # create line joining both points
    line = LineString([multipoint[0],multipoint[1]])

    # offset the line left and right by a distance equal to 10 times its length
    right_offset = line.parallel_offset(10*line.length, 'right')
    left_offset = line.parallel_offset(10*line.length, 'left')

    # find the centrepoints of the two offset lines
    right_centre = right_offset.interpolate(0.5, normalized=True)
    left_centre = left_offset.interpolate(0.5, normalized=True)

    # join the two centrepoints with a new line
    # this is the _perpendicular_bisector
    _perpendicular_bisector = LineString([left_centre, right_centre])

    return _perpendicular_bisector


def _explode_multipolygon(multipolygon):
    """
    separate a MultiPolygon into individual Polygons

    Parameters
    ----------
    multipolygon : MultiPolygon
        a shapely multipolygon

    Returns
    -------
    polygon_list : list
        a list of polygons
    """
    if not isinstance(multipolygon, MultiPolygon):
        print("input to '_explode_multipolygon was not a MultiPolygon")

    polygon_list = []

    # iterate through the geometry in the collection
    for shape in multipolygon:
        # separate out the polygons
        if isinstance(shape, Polygon):
            polygon_list.append(shape)
        # split any multipolygons into individual polygons - is this necessary?
        elif isinstance(shape, MultiPolygon):
            polygon_list += _explode_multipolygon(shape)
        else:
            print("split_multipolygons found something else")
    
    # check that polygons_list does indeed only contain polygons
    for polygon in polygon_list:
        if not isinstance(polygon, Polygon):
            print(f"split_mutipolygons: {type(polygon)} erroneously returned")

    return polygon_list


def _separate_fragments_from_polygons(polygons_, fraction=0.2):
    '''
    Separate a list of polygons into two lists of small and large polygons
    
    Parameters
    ----------
    polygons_ : list
        List of shapely Polygons
    fraction : float
        fraction of the median area to use to calculate the threshold size
    
    Returns
    -------
    fragments : List
        List of small shapely Polygons
    large_polygons : List
        List of large shapely Polygons
    '''
    polygon_areas = []
    fragments = []
    large_polygons = []
    
    # create a numpy array that holds the areas of all of the polygons
    for polygon_ in polygons_:
        if polygon_.area > 0:
            polygon_areas.append(polygon_.area)
    polygon_areas = np.array(polygon_areas)

    # calculate the median polygon area
    median_ = np.median(polygon_areas)

    # calculate threshold between small and large as fraction of median
    # e.g. 20% of the median
    threshold_ = median_*fraction

    # separate the polygons into fragments or large polygons depending on
    # whether they are smaller or larger than the threshold area
    for poly_ in polygons_:
        if poly_.area < threshold_:
            fragments.append(poly_)
        else:
            large_polygons.append(poly_)

    return fragments, large_polygons

def _merge_fragment_to_polygons(fragment, polygons):
    '''
    Helper function: Merge a single Polygon fragment onto one neighboring Polygon in a list of Polygons
    
    Parameter
    ---------
    fragment : Polygon
        single shapely Polygon
    polygons : list
        list of larger shapely Polygons
        
    Returns
    -------
    merged_polygons : list
        list of larger shapely Polygons including the newly merged Polygon
    '''
    #checks
    assert fragment.type == 'Polygon', "Fragment is not a Polygon"
    assert fragment.is_valid, "Fragment is not a valid Polygon"
    assert type(polygons) == list, "polygons is not a list"

    # make an empty list to hold the merged polygons
    merged_polygons = []

    # iterate through a copy (created by the slice) of the list of polygons
    # note: you can't iterate through a list you are simultaneously modifying
    for i, poly in enumerate(polygons[:]):
        if not poly.is_valid:
            print('poly is not valid')
        # if it intersects with the fragment along a line
        if type(poly.intersection(fragment)) == LineString:
            # make a new polygon union
            # check it is valid and append it to the merged polygon list
            # append all the other non-popped polygons to the merged list
            # stop the iteration
            new_poly = poly.union(fragment)
            if new_poly.is_valid:
                merged_polygons.append(new_poly)
                polygons.pop(i)
                merged_polygons += polygons
                break
            else:
                continue
        # if it doesn't intersect with the fragment
        else:
            continue

    if len(merged_polygons) <1:
        print("Fragment not merged", fragment)
        merged_polygons = polygons

    # return the new list
    return merged_polygons


def _merge_fragments_to_polygons(fragments, polygons):
    '''
    Merges a list of fragment Polygons onto a list of larger Polygons
    
    Each time a fragment is merged onto a larger Polygon the merged Polygon
    becomes part of the updated large Polygon list ready to receive the next
    fragment.
    
    Parameters
    ----------
    fragments : list
        list of small shapely Polygons
    polygons : list
        list of large shapely Polygons

    Returns
    -------
    merged_polygons : list
        list of shapely Polygons merged out of fragements and larger polygons
    '''
    # Create a new list of merged_polygons_ equal to the input polygons_
    merged_polygons = polygons
    
    # iterate through each fragment
    for fragment in fragments:
        # merging it on to one of the polygons, updating the merged_polygons list
        merged_polygons = _merge_fragment_to_polygons(fragment, merged_polygons)

    return merged_polygons


def _subdivide_polygon(idx, poly_, target_area_m2, voronoi_radius, points_per_cell, seed):
    '''
    subdivide a polygon using k-means clustering and voronoi regions
    
    Parameters
    ----------
    poly_: Polygon
        the polygon to subdivide
    target_area_m2: int
        Target area for subdivisions in square meters
    voronoi_radius: int
        The radius that PySAL will use to approximate infinity
    points_per_cell: int
        The approximate number of points in each cluster
    seed: int
        A random number seed for reproducibility

    Returns
    -------
    only_polygons_: list
        A list of shapely Polygons
    '''
    # Check the input is a single shapely Polygon
    if type(poly_) is not Polygon:
        raise TypeError ("'poly_ must be a single Shapely Polygon'")
    
    # If the input polygon is smaller than the threshold return it in a list
    if poly_.area <= target_area_m2:
        return [poly_]
    
    # fill the polygon with a MultiPoint object
    multipoint_ = _point_fill_poly(poly_,
                                   target_area_m2,
                                   points_per_cell,
                                   seed)

    # create cluster centroids
    cluster_centroids_ = _create_cluster_centroids(poly_, multipoint_, target_area_m2)

    # if one cluster is returned
    if len(cluster_centroids_)<=1:
        polygon_list_ = [poly_]

    # if 2 clusters bisect the shape
    elif len(cluster_centroids_)==2:
        # create a bisector line
        splitline_ = _perpendicular_bisector(cluster_centroids_)
        # split the polygon with the bisector
        polygon_list_ = split(poly_, splitline_)

    # else if clusters > 3 use voronoi technique
    elif len(cluster_centroids_)>2:
        # extract the coordinate of the centroid points
        centroid_coords = np.array(cluster_centroids_)

        # use PySAL's voronoi_frames to split the polygon based on the centroids
        # Note: if the radius is not adequate, first the shape will be 'trimmed', ultimately the dataframe will be returned empty
        # The radius is the distance that PySAL uses for points at infinity
        try:
            region_df, point_df = voronoi_frames(centroid_coords, voronoi_radius, clip=poly_)
            # check if the returned region_df is empty
            if region_df.empty:
                raise Exception
            # check if the returned region_df is smaller than 99.9% of the area of the original polygon
            # I use 99.9% as rounding errors(?) cause many cases to be very slightly smaller
            elif region_df.area.sum() < 0.999*poly_.area:
                raise Exception
        except Exception:
            print("The region_df returned by voronoi_frames for row {0} was either clipped or empty. Try increasing voronoi_radius".format(idx))

        # PySAL's voronoi_frames returns a GDF. Extract the geometry as a list of polys.
        polygon_list_ = list(region_df['geometry'])

    only_polygons=[]
    # if the split geometry contains multipolygons separate them into polygons
    for shape in polygon_list_:
        if isinstance(shape, MultiPolygon):
            exp_polys = _explode_multipolygon(shape)
            only_polygons += exp_polys
        elif isinstance(shape, Polygon):
            only_polygons.append(shape)

    # separate any fragments from the larger voronoi regions
    fragments_, polygons_ = _separate_fragments_from_polygons(only_polygons)

    # merge the fragments onto a neighbouring voronoi region
    only_polygons = _merge_fragments_to_polygons(fragments_, polygons_)

    return only_polygons


def _subdivide_row(idx, row, target_area_m2, voronoi_radius, points_per_cell, seed):
    '''
    Take a single GeoDataFrame row split its geometry, return a GDF of the result
    
    Parameters
    ----------
    idx : string or integer
        the id label of the row
    row : Series
        pandas series including geometry
    target_area_m2 : integer
        the targe size of the subdivided geometry
    voronoi_radius: int
        The radius that PySAL will use to approximate infinity
    points_per_cell: int
        The approximate number of points in each cluster
    seed: int
        A random number seed for reproducibility
    
    Returns
    -------
    row_gdf : GeoDataFrame
        a geodataframe of the subdivided geometry from that row with id numbers
    '''
    # list to hold subdivided polygons
    all_row_polys=[]

    # if the row geometry is a multipolygon
        # separate out the sub-polygons
        # then subdivide them
    # elif the row geometry is a polygon subdivide it directly
    if row['geometry'].type == 'MultiPolygon':
        polys_in = _explode_multipolygon(row['geometry'])
        for poly in polys_in:
            all_row_polys += _subdivide_polygon(idx,
                                                poly,
                                                target_area_m2,
                                                voronoi_radius,
                                                points_per_cell,
                                                seed)
    elif row['geometry'].type == 'Polygon':
        all_row_polys += _subdivide_polygon(idx,
                                            row['geometry'],
                                            target_area_m2,
                                            voronoi_radius,
                                            points_per_cell,
                                            seed)
    else:
        print('row contained geometry other than Polygons & MultiPolygons')
    
    # create a new GeoDataFrame from the list of polygons
    # Create a column labeled 'zona_id' with the original row id as a string
    # move the index numbers into a column
    # rename the new column 'index' to 'sub_zona_id'
    # ensure the 'sub_zona_id' values are strings
    row_gdf = gpd.GeoDataFrame(geometry=all_row_polys)
    row_gdf['zona_id'] = str(idx)
    row_gdf.reset_index(inplace=True)
    row_gdf.rename(columns={'index':'sub_zona_id'}, inplace=True)
    row_gdf['sub_zona_id'] = row_gdf['sub_zona_id'].astype(str)

    # reorder the columns
    row_gdf = row_gdf[['zona_id', 'sub_zona_id', 'geometry']]

    return row_gdf


def subdivide_zones(gdf, zone_id_col=None, target_area_m2=500000, voronoi_radius=15000000, points_per_cell=None, seed=None):
    '''
    Subdivide polygons in a GDF to a threshold size and return as a new GDF
    
    The function assumes the GDF is projected (coordinates in metres).
    If a column name is passed into unique_id_col the codes for the 
    generated zones will be based on this. Otherwise the index will be used.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        a GeoDataFrame with Polygon or MultiPolygon geometry
        must be projected and in metres
    unique_id_col : string
        column name of a column containing unique ID numbers
    target_area_m2 : int
        the targe size for the subdivisions
    voronoi_radius : int
        the distance used to approximate infinity in the voronoi function
    points_per_cell: int
        The approximate number of points in each cluster
    seed: int
        A random number seed for reproducibility
    
    Returns
    -------
    result_gdf : GeoDataFramea
        a new GeoDataFrame of zona subdivisions
    '''
    # List to hold all returned GDFs
    row_gdfs = []

    # Create a copy of the GDF so as not affect the original input
    gdf_in = gdf.copy()

    # For information print the input crs
    print("CRS of the input GDF:", gdf_in.crs, ". Assuming this is in metres.")
    
    # If a column of unique ID numbers was supplied make it the index
    # Check index numbers are unique in all cases i.e. if setting new index or not
    if zone_id_col is not None:
        gdf_in = gdf_in.set_index(zone_id_col)
    assert(gdf_in.index.is_unique), "The index or unique_id_col values are not unique."

    # iterate through each row of the input GDF and subdivide its geometry
    # append the returned row_gdf to the list of returned row_gdfs
    for idx, row in gdf_in.iterrows():
        ###print("id:", idx)                # <---print this to narrow down problem geometry
        row_gdf = _subdivide_row(idx,
                                 row,
                                 target_area_m2,
                                 voronoi_radius,
                                 points_per_cell,
                                 seed,
                                 )
        assert isinstance(row_gdf, gpd.GeoDataFrame), f"{idx} Didn't return a GeoDataFrame"
        row_gdfs.append(row_gdf)
        
    # Concatenate the individual row GDFs into a single GDF
    gdf_out = pd.concat(row_gdfs)

    # Find the maximum character length of the Zona_ID and Sub_Zona_ID
    max_len_zona_id = gdf_out['zona_id'].str.len().max()
    max_len_sub_zona_id = gdf_out['sub_zona_id'].str.len().max()
    # Pad the Zona & Sub_Zona_IDs with zeros to that length
    gdf_out['zona_id'] = gdf_out['zona_id'].str.zfill(max_len_zona_id)
    gdf_out['sub_zona_id'] = gdf_out['sub_zona_id'].str.zfill(max_len_sub_zona_id)
    # combine the zona_id onto the sub_zona_id
    gdf_out['sub_zona_id'] = gdf_out['zona_id'] + '-' + gdf_out['sub_zona_id']
    
    # sort the dataframe
    gdf_out.sort_values('sub_zona_id', inplace=True)
    gdf_out.reset_index(drop=True, inplace=True)
    # Set the crs for the new GDF from the crs of the original GDF
    gdf_out.crs = gdf.crs

    return gdf_out


def validate_geometry(gdf):
    '''
    Validates the geometry of a GeoDataFrame and buffers invalid geometries by 0

    Parameters
    ----------
    gdf : GeoDataFrame
        input GeoDataFrame of Polygonal geometry

    Returns
    -------
    gdf_ : GeoDataFrame
        output GeoDataFrame of Polygonal geometry
    '''

    # If not all geometry is valid
    if gdf.is_valid.sum() < len(gdf):
        
        # make a separate copy of the dataframe
        gdf_ = gdf.copy()

        # list the indices of the rows that have invalid geometry
        problem_row_ids = list(gdf_[~gdf_.is_valid].index)

        # find the index number of the 'geometry' column for .iloc
        geometry_col_id = [gdf_.columns.get_loc("geometry")]

        print(len(problem_row_ids), "invalid geometries found")

        print(gdf_.iloc[problem_row_ids, geometry_col_id])

        # Buffer the geometry of the problem rows
        gdf_.iloc[problem_row_ids, geometry_col_id] = gdf_.iloc[problem_row_ids, geometry_col_id].buffer(0)

        print("new geometries")
        print(gdf_.iloc[problem_row_ids, geometry_col_id])

        # Check if all geometry is valid
        print("All geometry is valid:", gdf_.is_valid.sum() == len(gdf_))

        return gdf_

    else:
        print("No invalid geometry found")

        return gdf


import pysal as ps
import warnings


def Superficie_dentro_de_Polígono(df_1, df_2, id1, id2):
    
    if 'Area_porc' in df_1.columns:
        del df_1['Area_porc']

    df1 = df_1.copy()
    df2 = df_2.copy()
    df1['Area_T'] = df1.area
    res_intersection = gpd.overlay(df1, df2, how='intersection')
    res_intersection['Area_P'] = res_intersection.area
    df1_Areadf2 = pd.DataFrame(res_intersection.groupby(id1)['Area_P'].sum()).reset_index()
    df1_T = pd.merge(df1, df1_Areadf2, left_on=id1, right_on=id1)
    df1_T['Area_porc'] = round(100 * df1_T['Area_P'] / df1_T['Area_T'], 2)

    r1 = pd.merge(df_1, df1_T[[id1, 'Area_porc']], how='left', on=id1)

    r1.loc[r1.Area_porc.isna(), 'Area_porc'] = 0

    return r1.sort_values(id1)


def creo_zonas(df_original, link, area_column, area_limit=1_000_000, link_agg=None, zona=None):
    """
    Agrega zonas juntando a los polígonos vecinos
     
    Requiere un campo ID (link) y un campo de area (area_column)
    
    area_limit es el area límite total por la que va a agregar polígonos. El proceso es iterativo y repite 
    al menos 4 iteraciones del dataframe, la primera agrega a un cuarto del límite de área y suma un cuarto las primeras 4 iteraciones.
    A partir de la 5ta iteración continúa iterando hasta que no quedan más polígonos posibles de agregar.
    Default de area_limit es 1km2

    link_agg=None : Todavía no está funcional

    Parameters
    ----------
    df_original: nombre del shape original que se quiere agregar
    link: link del shape original
    area_column: columna del área del polígono del shape original
    area_limit: tamaño de los nuevos polígonos
    link_agg:
    zona: nombre de la nueva zonificación (se le agrega el link del shape original)

    Returns
    -------
    Devuelve dos dfs:
        df original: con el agregado de un campo zona (si es None es 'zona_'+link)
        df_zona: nueva zonificación
    """

    warnings.filterwarnings('ignore')

    df_zona = df_original.copy()

    if zona==None:
        zona='zona_'+link

    it = 0
    area_limit_original = area_limit

    print('area_limit', area_limit)
    len_df = 0
    while (len_df != len(df_zona))|(it<4):
        it += 1

        if it < 4:
            area_limit = int((area_limit_original/4)*it)
        else:
            area_limit = area_limit_original
        print('Iteración:', it, 'Polígonos:', len(df_zona), 'Área límite:', area_limit)
        len_df = len(df_zona)

        df_zona=df_zona.sort_values(area_column).reset_index(drop=True)

        W = ps.lib.weights.contiguity.Queen.from_dataframe(df_zona, idVariable=link)  #trae los weights con los vecinos

        df_zona[zona] = 0
        nzone = 0

        df_zona['tiene_vecinos'] = 0

        for i, poly in df_zona.iterrows():    

#             if (i % 5000 == 0)&(i>0): print("   -----", i)

            # guarda el registro original y agrega a los vecinos
            neighbors = pd.DataFrame([])
            neighbors = neighbors.append(pd.DataFrame([[poly[link], poly[link], poly[area_column], '1_PPAL']], 
                                                      columns=[link, link+'_neigh', area_column, 'categoria']), ignore_index=True)
            for link_neigh in W[poly[link]]:
                neighbors = neighbors.append(pd.DataFrame([[poly[link], link_neigh, df_zona.loc[df_zona[link]==link_neigh, area_column].values[0], '2_NEIGH']], 
                                                          columns=[link, link+'_neigh', area_column, 'categoria']), ignore_index=True)

            # En los casos que pysal no encuentra vecinos (cuando un polígono está dentro de otro), busca vecinos con sjoin
            if len(neighbors)==1: 
                neighbors2 = df_zona.loc[df_zona[link].isin(neighbors[link+'_neigh'].to_list()),[link, 'geometry']]         
                n = gpd.sjoin(neighbors2[['geometry']], df_zona, op='intersects')[[link, area_column]].reset_index(drop=True)

                for _, link_neigh in n[n[link]!=poly[link]].iterrows():
                    neighbors = neighbors.append(pd.DataFrame([[poly[link], link_neigh[link], df_zona.loc[df_zona[link]==link_neigh[link], area_column].values[0],
                                                                '2_NEIGH']], columns=[link, link+'_neigh', area_column, 'categoria']), ignore_index=True)

            neighbors = df_zona.loc[df_zona[link].isin(neighbors[link+'_neigh'].to_list()),[link, 'geometry']]
            neighbors['distancia_neigh']= neighbors.centroid.distance(poly.geometry.centroid)
            neighbors = neighbors.sort_values('distancia_neigh').reset_index(drop=True)

            if len(neighbors)>1: df_zona.loc[df_zona[link]==poly[link], 'tiene_vecinos'] += 1

            # Agrega número de zona
            if (df_zona.loc[df_zona[link]==poly[link], zona].values[0]==0):

                nzone += 1
                df_zona.loc[df_zona[link] == poly[link], zona] = nzone
                area_sum = 0

                for ii_neigh, df_neigh in neighbors.iterrows():

                    if (((df_zona.loc[df_zona[link]==df_neigh[link], zona].values[0]==0)|(df_zona.loc[df_zona[link]==df_neigh[link], zona].values[0]==nzone))&
                        (area_limit >= (df_zona.loc[df_zona[link]==df_neigh[link], area_column].values[0]+area_sum))):

                        df_zona.loc[df_zona[link]==df_neigh[link], zona]=nzone
                        area_sum += df_zona.loc[df_zona[link]==df_neigh[link], area_column].values[0]

        df_zona = df_zona[[zona, area_column, 'tiene_vecinos', "geometry"]].dissolve(zona, aggfunc='sum').reset_index()

        df_zona = df_zona.rename(columns={zona:link})


    df_zona = df_zona.rename(columns={link:zona})

    df_zona.loc[df_zona['tiene_vecinos']>0, 'tiene_vecinos'] = 1

    print('Cantidad de iteraciones:', it)

    df_zona[zona] = df_zona[zona].astype(str).str.zfill(6)

    llen = len(df_original)

    df_original_temp = df_original[[link, 'geometry']]
    df_original_temp['geometry'] = df_original_temp['geometry'].representative_point()

    df_original_temp = gpd.sjoin(df_original_temp, df_zona[[zona, 'tiene_vecinos', 'geometry']], op='within', how='left')

    df_original = df_original.set_index(link).join(df_original_temp[[link, zona, 'tiene_vecinos']].set_index(link)).reset_index()

    df_original['csum'] = df_original.groupby(link).cumcount()
    df_original = df_original[df_original.csum==0].reset_index(drop=True)
    del df_original['csum']

    if len(df_original)!= llen: print('ERROR: Cuidado, cambio la cantidad de registros en el dataframe original')

    if len(df_zona[df_zona[zona].isna()]) > 0: print('ERROR: Cuidado, hay zonas con NULL -- Revisar', len(df_zona[df_zona[zona].isna()]))

    del df_original['tiene_vecinos']
    del df_zona['tiene_vecinos']

    df_zona = df_original.dissolve(zona, aggfunc='sum').reset_index()

    return df_original, df_zona