# Facón

Facón is an experimental python library for subdividing a GeoDataFrame of administrative boundaries into a new GeoDataFrame of approximately equal area zones with evenly distributed centres constrained by the boundaries of the original geometries. It is intended to work with projected administrative areas with their coordinates in metres. A target area in metres squared is used to determine how many subdivisions should be created. It was developed in collaboration with Sebastián Anapolsky [@sanapolsky](https://github.com/sanapolsky/).

## Working with subdivisions

Administrative boundaries are typically driven by a need to maintain an approximately equal population in each area while maintaining spatial contiguity (i.e. no gaps between the zones). These constraints can lead to administrative areas that vary greatly in area (m2).

![Greater London boroughs, Ciudad Autónoma de Buenos Aires comunas](https://github.com/AtelierLibre/AtelierLibre.github.io/blob/master/images/London_boroughs_Buenos_Aires_comunas.png) "Greater London boroughs, Ciudad Autónoma de Buenos Aires comunas")

Transport analysis in contrast tends to take a regular (square or hexagonal) grid as its starting point, creating a set of zones of equal area with an equal distance from the centre of one zone to the centre of the next.

These two approaches to subdivisions conflict when, for example, carrying out studies into the relationship between social indicators from census data and the time/cost of travel.

## A hybrid solution

Facón proposes a hybrid solution, creating approximately equal area zones with approximately evenly spaced centres that nevertheless respect the original administrative boundaries. It uses K-Means clustering to define the subdivisions and their centroids and voronoi regions to create the boundaries. It benefits from the fact that K-Means creates a predefined number of clusters (here determined by dividing the area (m2) of the administrative zone by the desired target area (m2) of the new zones) that are approximately circular.

![Greater London boroughs, Ciudad Autónoma de Buenos Aires comunas subdivided](https://github.com/AtelierLibre/AtelierLibre.github.io/blob/master/images/London_boroughs_Buenos_Aires_comunas_subdivided.png) "Greater London boroughs, Ciudad Autónoma de Buenos Aires comunas subdivided")

## References

It was inspired by two prior pieces of work:

### [Geovoronoi by Markus Konrad at WZB (Berlin Social Science Center)](https://github.com/WZBSocialScienceCenter/geovoronoi)

Geovoronoi is a python library for generating voronoi regions inside an administrative boundary given a set of predetermined starting points. The regions that it generates are driven by a set of given centres and are not necessarily of similar area.

### [Paul Ramsey - PostGIS Polygon Splitting](http://blog.cleverelephant.ca/2018/06/polygon-splitting.html)

This post by Paul Ramsey on achieving the desired result with PostGIS was instrumental in pointing the way to using K-Means clustering to subdivide the polygons into approximately equal areas.
