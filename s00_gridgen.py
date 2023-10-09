import numpy as np
import pandas as pd
import flopy
import geopandas as gpd
import os
from flopy.utils.gridgen import Gridgen
import matplotlib.pyplot as plt
import shutil
from shapely.geometry import Point

print(f'using special version of flopy with more freedom in gridgen\n{flopy.__path__}')

def get_OG_grid_info():
    grid_feet = gpd.read_file(os.path.join('GIS', 'input_shapefiles', 'grid_feet_points.shp'))
    top_bot = gpd.read_file(os.path.join('GIS', 'input_shapefiles', 'OAHU_USG_Grid_1d.shp'))

    # # for lay in range(25):
    # #     grid_df = grid_feet[grid_feet['layer'] == lay+1]
    # #     tb = top_bot[top_bot['layer'] == lay+1]
    # shp_joined = gpd.sjoin_nearest(grid_feet, top_bot, on='layer')

    nlay = 25
    nrow = max(grid_feet.row)
    ncol = max(grid_feet.column)
    delr, delc = 1574.80316230315, 1574.80316230315
    xoff, yoff = 2002504, 7715906
    rotation = 40
    proj4 = '+proj=tmerc +lat_0=0 +lon_0=-159 +k=0.9996 +x_0=500000.001016002 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs'

    grid_feet_df = pd.DataFrame(grid_feet.drop(columns='geometry'))

    mf = flopy.mfusg.MfUsg.load('oahu.nam', model_ws=os.path.join('model_ws', 'oahu_ft_00', 'model2'),
                                    version='mfusg', check=False, forgive=True, load_only=['disu','lpf','bas6'])
    nodelay = [10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268,
               10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268]

    nodes = sum(nodelay)
    print('Number of New Nodes ', nodes)

    old_bas = mf.bas6
    ibound = old_bas.ibound.array

    old_disu = mf.disu

    # old_top, old_botm = old_disu.top.array, old_disu.bot.array
    old_top = top_bot['top']
    old_botm = top_bot['bottom']
    old_thickness = old_top - old_botm
    old_area = old_disu.area.array

    print(len(old_thickness))
    old_thickness = old_thickness[old_thickness > 0]
    print(len(old_thickness))
    print(old_disu)
    print(old_disu.iac.array.shape)

    # grid_feet_df['top'] = old_top
    # grid_feet_df['bottom'] = old_botm
    # grid_feet_df['thickness'] = old_thickness

    node_df = pd.DataFrame(columns = ['nodenumber', 'top', 'bottom', 'thickness', 'area'])
    node_df['nodenumber'] = top_bot['nodenumber']
    node_df['top'], node_df['bottom'], node_df['thickness'], node_df['area'] = old_top, old_botm, old_thickness, old_area
    # areas = grid_feet_df['area'].unique()
    # # areas_disu = np.unique(old_area)
    # for i in range(len(grid_feet_df)):
    #     if grid_feet_df['area'][i] > 230000:
    #         grid_feet_df['area'][i] = 2480005.0
    #     if 50000 < grid_feet_df['area'][i] <60000:
    #         grid_feet_df['area'][i] = 620001.3
    #     if 10000 < grid_feet_df['area'][i] < 15000:
    #         grid_feet_df['area'][i] = 155000.3
    #     if grid_feet_df['area'][i] < 4000:
    #         grid_feet_df['area'][i] = 38750.08

    # grid_feet_df['area'] = old_area

    grid_feet_df_final = grid_feet_df.merge(node_df, left_on='nodenumber', right_on='nodenumber')
    grid_feet_df_final.drop(columns=['area_x', 'node', 'zone'], inplace=True)
    grid_feet_df_final.rename(columns={"area_y": "area"}, inplace=True)
    grid_feet_df_final = grid_feet_df_final.sort_values(by='nodenumber')
    grid_feet_df_final.to_csv(os.path.join('input_data', 'node.info'), index=False, sep ='\t')

    return nlay, nrow, ncol, delr, delc, xoff, yoff, rotation, proj4

def save_layer_footprint():
    nlay = 25
    proj4 = '+proj=tmerc +lat_0=0 +lon_0=-159 +k=0.9996 +x_0=500000.001016002 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs'

    grid_feet = gpd.read_file(os.path.join('GIS', 'input_shapefiles', 'grid_feet_projected.shp'))
    # proj4 = '+proj=aea +lat_1=27.5 +lat_2=35 +lat_0=31.25 +lon_0=-100 +x_0=1500000 +y_0=6000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs'
    usg_shp = grid_feet.to_crs(proj4)
    lay_fps_dir = os.path.join('GIS','input_shapefiles')

    temp_shp = usg_shp.loc[usg_shp['layer'] == 1]
    temp_shp['geometry'] = temp_shp['geometry'].buffer(0.0001)
    temp_shp['share'] = 1
    temp_shp = temp_shp.dissolve('share')
    # temp_shp['geometry'] = temp_shp['geometry'].buffer(-0.0001)
    temp_shp['geometry'] = temp_shp['geometry'].buffer(-10)

    temp_shp.to_file(os.path.join(lay_fps_dir,'layer_footprint.shp'))

def get_botms(top,nlay,nrow,ncol, plot=False):
    grid_feet = gpd.read_file(os.path.join('GIS', 'input_shapefiles', 'grid_feet_points.shp'))
    grid_feet = grid_feet.sort_values(by='nodenumber')
    nlay = 25
    nrow = max(grid_feet.row)
    ncol = max(grid_feet.column)
    botm = np.ones((nlay,nrow,ncol)) * -9999

    node_info = pd.read_csv(os.path.join('input_data','node.info'),delim_whitespace=True)
    for lay in range(nlay):
        temp_df = node_info.loc[node_info['layer'] == lay+1]
        botm[lay][temp_df['row']-1, temp_df['column']-1] = temp_df['bottom']

    # for lay in range(nlay):
    #     if lay > 0:
    #         botm[lay][botm[lay] == -9999] = botm[lay-1][botm[lay] == -9999]
    #     else:
    #         botm[lay][botm[lay] == -9999] = top[botm[lay] == -9999]


    if plot:
        for lay in range(nlay):
            botm[botm == -9999] = np.nan
            # botm[botm != -9999] = np.nan

            fig, ax = plt.subplots(figsize=(8,6))
            plt.imshow(botm[lay],cmap='jet')
            ax.set_title(f'bottom lay {lay+1}')
            plt.colorbar()
        plt.show()

    return botm

def get_tops(nrow,ncol,plot=False):
    grid_feet = gpd.read_file(os.path.join('GIS', 'input_shapefiles', 'grid_feet_points.shp'))
    grid_feet = grid_feet.sort_values(by='nodenumber')
    nlay = 25
    nrow = max(grid_feet.row)
    ncol = max(grid_feet.column)
    top = np.ones((nrow,ncol)) * -9999
    # top = np.ones((nlay, nrow, ncol)) * -9999

    node_info = pd.read_csv(os.path.join('input_data','node.info'),delim_whitespace=True)
    node_info['bottom'] = -node_info['bottom']
    node_info = pd.DataFrame(node_info.groupby(['layer','row','column'])['top', 'bottom'].max()).reset_index()
    node_info['bottom'] = -node_info['bottom']
    node_info.drop_duplicates(['row','column'],keep='first',inplace=True)
    # node_info = node_info[node_info['layer'] == 1]


    top[node_info['row']-1, node_info['column']-1] = node_info['top']

    if plot:
        fig, ax = plt.subplots(figsize=(8,6))
        # top[top <= top.min()] = np.nan
        plt.imshow(top,cmap='jet')

        plt.colorbar()
        plt.show()


    return top

def create_coarseGrid(nrow, ncol, nlay, delr, delc, xoff, yoff, rotation, proj4):
    grid_feet = gpd.read_file(os.path.join('GIS', 'input_shapefiles', 'grid_feet_points.shp'))

    nlay = 25
    nrow = max(grid_feet.row)
    ncol = max(grid_feet.column)
    delr, delc = 1574.80316230315, 1574.80316230315
    xoff, yoff = 2002504, 7715906
    rotation = 40
    proj4 = '+proj=tmerc +lat_0=0 +lon_0=-159 +k=0.9996 +x_0=500000.001016002 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs'

    # ref_cell_df = pd.read_csv(os.path.join('input_data', 'GWV_maps', 'brazos_colorados_cells_plusVistaRidge.map'),
    #                           skiprows=1, names=['x1', 'y1', 'x2', 'y2', '1', '3'], delim_whitespace=True)
    #
    # ref_cell_df = get_refined_cells()
    # proj4 = '+proj=aea +lat_1=27.5 +lat_2=35 +lat_0=31.25 +lon_0=-100 +x_0=1500000 +y_0=6000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs'

    mfs = flopy.modflow.Modflow('RH_struct', 'RH_struct.nam', version='mfnwt',
                                model_ws=os.path.join('model_ws', 'RH_struct'), rotation=rotation, xll=xoff, yll=yoff,
                                proj4_str=proj4)

    top = 10000
    botm = -10
    botm = -np.arange(0, nlay)

    top = get_tops(nrow, ncol)
    botm = get_botms(top, nlay, nrow, ncol)

    dis = flopy.modflow.ModflowDis(mfs,
                                   nlay=nlay,
                                   nrow=nrow,
                                   ncol=ncol,
                                   delr=delr,
                                   delc=delc,
                                   top=top,
                                   botm=botm,
                                   )

    # mfs.modelgrid.set_coord_info(xoff,yoff,rotation,proj4=proj4)

    gridgen_exe = os.path.join('bin', 'windows', "gridgen.exe")
    gridgen_ws = os.path.join('model_ws', 'RH_gridgen_ws')

    surface_interpolation = {lay: 'replicate' for lay in range(nlay + 1)}
    surface_interpolation[0] = 'interpolate'
    surface_interpolation[1] = 'interpolate'

    g = Gridgen(dis, model_ws=gridgen_ws, exe_name=gridgen_exe, surface_interpolation=surface_interpolation,
                vertical_pass_through=True)

    g.build(verbose=True)
    print('finished first --------')

def get_refined_cells():
    delr, delc = 1574.80316230315, 1574.80316230315

    cell2refine = {round(delr ** 2, 1): 0,
                   round((delr / 2) ** 2, 1): 1,
                   round((delr / 4) ** 2, 1): 2,
                   round((delr / 8) ** 2, 2): 3,
                   round((delr / 16) ** 2, 2): 4,
                   round((delr / 32) ** 2, 2): 5}

    # cell2refine = {round(delr ** 2, 1): 0,
    #                round((delr / 2) ** 2, 1): 1,
    #                round((delr / 4) ** 2, 1): 2,
    #                round((delr / 8) ** 2, 2): 3}

    node_info = pd.read_csv(os.path.join('input_data','node.info'),delim_whitespace=True)
    # node_info = node_info.loc[node_info['layer'] == 10]
    # node_info = pd.DataFrame(node_info.groupby(['row','column'])['x','y','area'].mean().reset_index(drop=True))
    # node_info.drop_duplicates(['row','column'],inplace=True)
    node_info['refine'] = node_info.apply(lambda i: cell2refine[i['area']],axis=1)



    return node_info

def read1d(f, a):
    """
    Quick file to array reader for reading gridgen output.  Much faster
    than the read1d function in util_array
    """
    dtype = a.dtype.type
    lines = f.readlines()
    l = []
    for line in lines:
        l += [dtype(i) for i in line.strip().split()]
    a[:] = np.array(l, dtype=dtype)
    return a

def update_disu_props(test=True):
    # def distance(x1,y1,x2,):
    #     return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    gridgen_ws = os.path.join('model_ws', 'RH_gridgen_ws')
    grid_df = pd.read_csv(os.path.join('input_data', 'rf.node.lookup.csv'))
    new2old, old2new = get_new2old(grid_df)

    ognode_info = pd.read_csv(os.path.join('input_data','node.info'),delim_whitespace=True)
    ognode_info['lrc'] = ognode_info.apply(lambda i: f'{i["layer"]}_{i["row"]}_{i["column"]}', axis=1)

    ogdata = []
    for row in ognode_info.values:
        ognode = row[3]
        nnodes = old2new[ognode]
        for nnode in nnodes:
            ogdata.append(list(row)+[nnode])


    ognode_info = pd.DataFrame(ogdata,columns=list(ognode_info.columns)+['newnode'])
    ognode_info.sort_values('newnode',inplace=True) # because the og grid was done by vistas and the new grid was in gridgen... they doe quad tree numbering different... sigh

    nodelay_fn = open(os.path.join(gridgen_ws,'qtg.nodesperlay.dat'))
    nodelay = [int(node) for node in nodelay_fn.readlines()[0].split()]
    nodes = np.sum(nodelay)
    nlay = len(nodelay)

    top = []
    bot = []
    z = []
    area = [float(v) for line in open(os.path.join(gridgen_ws,f'qtg.area.dat')).readlines() for v in line.split()]
    ja = [int(v) for line in open(os.path.join(gridgen_ws, f'qtg.ja.dat')).readlines() for v in line.split()]
    fahl = [float(v) for line in open(os.path.join(gridgen_ws, f'qtg.fahl.dat')).readlines() for v in line.split()]
    cl12 = [float(v) for line in open(os.path.join(gridgen_ws, f'qtg.c1.dat')).readlines() for v in line.split()]
    layer = []

    for lay in range(nlay):
        nn = nodelay[lay]
        a = np.empty((nn), dtype=np.float32)

        top += [float(v) for line in open(os.path.join(gridgen_ws,f'quadtreegrid.top{lay+1}.dat')).readlines() for v in line.split()]
        bot += [float(v) for line in open(os.path.join(gridgen_ws,f'quadtreegrid.bot{lay+1}.dat')).readlines() for v in line.split()]
        layer += [lay+1]*nn

    qtgrid = gpd.read_file(os.path.join(gridgen_ws,'qtgrid.shp'))
    row = (qtgrid['row']+1).astype(int).tolist()
    column = (qtgrid['col']+1).astype(int).tolist()


    data = {'nodenumber':np.arange(0,nodes)+1,
            'layer':layer,
            'row':row,
            'column':column,
            'area':area,
            'top':top,
            'bottom':bot}

    df = pd.DataFrame(data)
    df['z'] = (df['top'] + df['bottom'])/2
    # now lets fix the tops and bottoms
    df['ognodenumber'] = [new2old[node] for node in df['nodenumber'].tolist()]
    # df.sort_values('ognodenumber',inplace=True) # because the og grid was done by vistas and the new grid was in gridgen... they doe quad tree numbering different... sigh
    df['lrc'] = df.apply(lambda i: f'{i["layer"]}_{i["row"]}_{i["column"]}', axis=1)
    lrcs = df['lrc'].unique()

    newtop, newbot= [], []
    for lrc in lrcs:
        # print(lrc)
        lay = int(float(lrc.split('_')[0]))
        if lay <= 7:
            temp_df = df.loc[df['lrc'] == lrc]
            temp_ogdf = ognode_info.loc[ognode_info['lrc'] == lrc]

            if (len(temp_df) == 1) and (temp_df['area'].mean() == df['area'][0]): # no refinement, no problem
                newtop += temp_df['top'].tolist()
                newbot += temp_df['bottom'].tolist()

            elif len(temp_df) == len(temp_ogdf): # no refinement, no problem
                if lay == 1: # layer 1, use old grid refinement
                    newtop += temp_ogdf['top'].tolist()
                    newbot += temp_ogdf['bottom'].tolist()
                    # if len(temp_ogdf) > 4:
                    #     print('oy')
                elif lay == 2: #layer 2 top, use old grid refinement
                    newtop += temp_ogdf['top'].tolist()
                    newbot += temp_ogdf['bottom'].tolist() # new

                else:
                    newtop += temp_ogdf['top'].tolist() # new
                    newbot += temp_ogdf['bottom'].tolist() # new
            else:
                p = 'fuck'
                print(lrc, p)
        else:
            break
    #
    # # for lay in [3,4,5,6,7,8,9,10]:
    for lay in [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]:
        temp_df = df.loc[df['layer'] == lay]
        newtop += temp_df['top'].tolist()  # new
        newbot += temp_df['bottom'].tolist()  # new
    #
    # print(len(newtop),len(newbot))
    #
    #
    qtgrid['top'] = newtop
    qtgrid['bottom'] = newbot

    qtgrid.to_file(os.path.join(gridgen_ws,'qtgrid.shp'))

    df['top'] = newtop
    df['bottom'] = newbot
    df['z'] = (df['top'] + df['bottom'])/2

    node_info_dict = df.set_index('nodenumber').to_dict()

    if not test:
        for lay in range(nlay):
            topl = df['top'].loc[df['layer'] == lay+1].tolist()
            np.savetxt(os.path.join(gridgen_ws,f'quadtreegrid.top{lay+1}.dat'),topl)

            botl = df['bottom'].loc[df['layer'] == lay+1].tolist()
            np.savetxt(os.path.join(gridgen_ws,f'quadtreegrid.bot{lay+1}.dat'),botl)

    print('eh')
    # model_ws = os.path.join('model_ws','rf.base.wel.fix.1929-2010.pst18')
    # mf = flopy.mfusg.MfUsg.load('rf.gma12.nam',model_ws=model_ws,load_only=['disu'],forgive=True,check=False)

    # disu = mf.disu


    # iac : array of integers
    #     is a vector indicating the number of connections plus 1 for each
    #     node. Note that the IAC array is only supplied for the GWF cells;
    #     the IAC array is internally expanded to include CLN or GNC nodes if
    #     they are present in a simulation.
    #     (default is None. iac must be provided).
    # iac = disu.iac.array
    iac = [int(v) for line in open(os.path.join(gridgen_ws,f'qtg.iac.dat')).readlines() for v in line.split()]

    # ja : array of integers
    #     is a list of cell number (n) followed by its connecting cell numbers
    #     (m) for each of the m cells connected to cell n. This list is
    #     sequentially provided for the first to the last GWF cell. Note that
    #     the cell and its connections are only supplied for the GWF cells and
    #     their connections to the other GWF cells. This connectivity is
    #     internally expanded if CLN or GNC nodes are present in a simulation.
    #     Also note that the JA list input may be chopped up to have every node
    #     number and its connectivity list on a separate line for ease in
    #     readability of the file. To further ease readability of the file, the
    #     node number of the cell whose connectivity is subsequently listed,
    #     may be expressed as a negative number the sign of which is
    #     subsequently corrected by the code.
    #     (default is None.  ja must be provided).
    # ja = disu.ja.array
    ja = [int(v) for line in open(os.path.join(gridgen_ws,f'qtg.ja.dat')).readlines() for v in line.split()]


    # fahl : float or arry of floats
    #     Area of the interface Anm between nodes n and m.
    #     (default is None.  fahl must be specified.)
    # _fahl = disu.fahl.array
    _fahl = [float(v) for line in open(os.path.join(gridgen_ws,f'qtg.fahl.dat')).readlines() for v in line.split()]
    if not test:
        shutil.copy2(os.path.join(gridgen_ws,'qtg.fahl.dat'),os.path.join(gridgen_ws,'qtg.fahl.b4fix.dat'))

    #  cl12 : float or array of floats
    #     is the array containing CL1 and CL2 lengths, where CL1 is the
    #     perpendicular length between the center of a node (node 1) and the
    #     interface between the node and its adjoining node (node 2). CL2,
    #     which is the perpendicular length between node 2 and the interface
    #     between nodes 1 and 2 is at the symmetric location of CL1. The array
    #     CL12 reads both CL1 and CL2 in the upper and lower triangular
    #     portions of the matrix respectively. Note that the CL1 and CL2 arrays
    #     are only supplied for the GWF cell connections and are internally
    #     expanded if CLN or GNC nodes exist in a simulation.
    #     (default is None.  cl1 and cl2 must be specified, or cl12 must be
    #     specified)
    # _cl12 = disu.cl12.array
    _cl12 = [float(v) for line in open(os.path.join(gridgen_ws,f'qtg.c1.dat')).readlines() for v in line.split()]
    if not test:
        shutil.copy2(os.path.join(gridgen_ws,'qtg.c1.dat'),os.path.join(gridgen_ws,'qtg.c1.b4fix.dat'))

    cl1f = open(os.path.join(gridgen_ws,'qtg.c1.dat'),'w')
    fahlf = open(os.path.join(gridgen_ws,'qtg.fahl.dat'),'w')



    idx = 0
    for node in range(nodes):
        n = iac[node]
        cells = ja[idx:idx+n]
        lay = node_info_dict['layer'][node+1]
        cl12_row = []
        fahl_row = []
        for cell in cells:

            if cell == node+1: #same cell
                dist = 0
                a = 0

            elif lay == node_info_dict['layer'][cell]: # same layer
                dist = int(np.sqrt(node_info_dict['area'][node+1])/2)
                avg_thk = np.average([node_info_dict['top'][node+1] - node_info_dict['bottom'][node+1],node_info_dict['top'][cell] - node_info_dict['bottom'][cell]])

                # area should be the smaller of the two area if cells are different sizes
                a = np.min([np.sqrt(node_info_dict['area'][node+1]) * avg_thk, np.sqrt(node_info_dict['area'][cell]) * avg_thk])

            elif lay < node_info_dict['layer'][cell]:
                dist = node_info_dict['z'][node+1] - node_info_dict['top'][cell]
                a = int(node_info_dict['area'][node+1])

            elif lay > node_info_dict['layer'][cell]:
                dist = node_info_dict['bottom'][cell] - node_info_dict['z'][node + 1]
                a = int(node_info_dict['area'][node+1])

            cl12_row.append(dist)
            fahl_row.append(a)


            # print('h')
        cl1f.write('{}\n'.format(' '.join([str(x) for x in cl12_row])))
        fahlf.write('{}\n'.format(' '.join([str(x) for x in fahl_row])))

        idx+=n

def gen_usg_model(plot=True):
    #
    # ref_cell_df = pd.read_csv(os.path.join('input_data','GWV_maps','brazos_colorados_cells_plusVistaRidge.map'),
    #                            skiprows=1,names=['x1','y1','x2','y2','1','3'], delim_whitespace=True)

    grid_feet = gpd.read_file(os.path.join('GIS', 'input_shapefiles', 'grid_feet_points.shp'))

    nlay = 25
    nrow = max(grid_feet.row)
    ncol = max(grid_feet.column)
    delr, delc = 1574.80316230315, 1574.80316230315
    xoff, yoff = 2002504, 7715906
    rotation = 40
    proj4 = '+proj=tmerc +lat_0=0 +lon_0=-159 +k=0.9996 +x_0=500000.001016002 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs'

    ref_cell_df = get_refined_cells()

    mfs = flopy.modflow.Modflow('RH_struct', 'RH_struct.nam', version='mfnwt',
                                model_ws=os.path.join('model_ws', 'RH_struct'), rotation=rotation, xll=xoff, yll=yoff,
                                proj4_str=proj4)

    nodelay = [10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268,
               10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268, 10268]


    top = get_tops(nrow, ncol,plot=False)
    botm = get_botms(top, nlay, nrow, ncol,plot=False)

    dis = flopy.modflow.ModflowDis(mfs,
                                   nlay=nlay,
                                   nrow=nrow,
                                   ncol=ncol,
                                   delr=delr,
                                   delc=delc,
                                   top=top,
                                   botm=botm,
                                   )

    # mfs.modelgrid.set_coord_info(xoff,yoff,rotation,proj4=proj4)

    gridgen_exe = os.path.join('bin', 'windows', "gridgen.exe")
    gridgen_ws = os.path.join('model_ws', 'RH_gridgen_ws')

    surface_interpolation = {lay: 'replicate' for lay in range(nlay + 1)}
    surface_interpolation[0] = 'interpolate'
    surface_interpolation[1] = 'interpolate'

    g = Gridgen(dis, model_ws=gridgen_ws, exe_name=gridgen_exe, surface_interpolation=surface_interpolation,
                vertical_pass_through=True)


    for ref in [1,2, 3]:
        points = list(zip(ref_cell_df['x'].loc[ref_cell_df['refine'] == ref].tolist(), ref_cell_df['y'].loc[ref_cell_df['refine'] == ref].tolist()))

        g.add_refinement_features(points, "point", ref, range(nlay))


    refine_shps = {'shaft_refinement_poly': 5,
                   'tanks_wells_refinement': 4}
    refine_shps_geoms = {'shaft_refinement_poly': 'polygon',
                         'tanks_wells_refinement': 'polygon'}

    for shp in refine_shps.keys():
        ref = refine_shps[shp]
        geom = refine_shps_geoms[shp]
        print(f'refining {shp} at level {ref} type {geom}')

        g.add_refinement_features(os.path.join(os.getcwd(),'GIS','input_shapefiles',shp), geom, ref, range(nlay))

    lay_fps_dir = os.path.join(os.getcwd(), 'GIS','input_shapefiles')
    for lay in np.arange(nlay):
        active_shp = os.path.join(lay_fps_dir,'layer_footprint')

        g.add_active_domain(active_shp, [lay])

    g.build(verbose=True)
    print('finished first --------')

    gen_oldnode2newnode2(proj4) # get lookup table for old2new node and new2old
    update_disu_props(test=False)
    print('finsihed updating tops, botms, cl12 and fahl')
    mu = flopy.mfusg.MfUsg(model_ws=gridgen_ws, modelname="mfusg")

    nper = 118
    perlen, steady = [365.25]*nper, [False]*nper
    perlen[0]=3652.5
    steady[0] = False
    disu = g.get_disu(mu,nper=nper,perlen=perlen,nstp=1,tsmult=1.2,steady=steady, lenuni=1)
    disu.write_file()
    print('finsihed building disu')

    if plot:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        mm = flopy.plot.PlotMapView(model=mfs)
        mm.plot_grid()
        ax.scatter(ref_cell_df['x'],ref_cell_df['y'],color='r')
        # ax.scatter(ref_cell_df['x2'],ref_cell_df['y2'],color='b')

        # flopy.plot.plot_shapefile(rf2shp, ax=ax, facecolor="yellow", edgecolor="none")
        # flopy.plot.plot_shapefile(rf1shp, ax=ax, linewidth=10)
        # flopy.plot.plot_shapefile(rf0shp, ax=ax, facecolor="red", radius=1)
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        g.plot(ax, linewidth=.5)
        ax.scatter(ref_cell_df['x'],ref_cell_df['y'],color='r')

        plt.show()

def gen_new_node_info():
    # proj4 = '+proj=aea +lat_1=27.5 +lat_2=35 +lat_0=31.25 +lon_0=-100 +x_0=1500000 +y_0=6000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs'

    proj4 = '+proj=tmerc +lat_0=0 +lon_0=-159 +k=0.9996 +x_0=500000.001016002 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs'

    gridgen_ws = os.path.join('model_ws','RH_gridgen_ws')

    qtgrid = gpd.read_file(os.path.join(gridgen_ws,'qtgrid.shp'))
    qtgrid.crs = proj4
    qtgrid['layer'] = qtgrid['layer'].astype(int)+1
    qtgrid['aquifer'] = qtgrid['layer']
    qtgrid['area'] = round(qtgrid['geometry'].area,0).astype(int)

    qtgrid['geometry'] = qtgrid['geometry'].centroid

    # qtgrid['x'] = qtgrid.apply(lambda i: format(i["geometry"].x, ".3f"), axis=1)
    # qtgrid['y'] = qtgrid.apply(lambda i: format(i["geometry"].y, ".3f"), axis=1)

    qtgrid['x'] = [format(geom.x,'.3f') for geom in qtgrid['geometry'].tolist()]
    qtgrid['y'] = [format(geom.y,'.3f') for geom in qtgrid['geometry'].tolist()]

    qtgrid['row'] = qtgrid['row'].astype(int)+1
    qtgrid['column'] = qtgrid['col'].astype(int)+1

    # qtgrid['top'] = round(qtgrid['top'],3)
    # qtgrid['bottom'] = round(qtgrid['bottom'],3)
    qtgrid['z'] = (qtgrid['top'] + qtgrid['bottom'])/2
    # qtgrid['z'] = (qtgrid['top'] - qtgrid['bottom'])/1

    qtgrid['top'] = [format(col,'.3f') for col in qtgrid['top'].tolist()]
    qtgrid['bottom'] = [format(col,'.3f') for col in qtgrid['bottom'].tolist()]
    qtgrid['z'] = [format(col,'.3f') for col in qtgrid['z'].tolist()]


    qtgrid['outcrop'] = qtgrid['layer']
    qtgrid['flag'] = qtgrid['layer']
    qtgrid['paired_node'] = 0

    node_info = qtgrid[['nodenumber','layer','aquifer','row','column','x','y','z','area','top','bottom','outcrop','flag','paired_node']]

    node_info['geometry'] = [Point(pts) for pts in list(zip(node_info['x'].astype(float).tolist(),node_info['y'].astype(float).tolist()))]
    node_gdf = gpd.GeoDataFrame(node_info,crs=proj4)

    usg_shp = gpd.read_file(os.path.join('GIS', 'input_shapefiles', 'grid_feet_projected.shp'))
    # node_gdf = gpd.sjoin(node_gdf,usg_shp[[f'nodeL{lay + 1}' for lay in range(nlay)]+['layer1ID','layer2ID','geometry']])
    node_gdf = gpd.sjoin(node_gdf, usg_shp[[f'nodenumber'] + ['layer', 'geometry']])
    node_gdf['layer_right'].loc[np.isnan(node_gdf['layer_right'])] = 0

    # lay2aq = {int(val[0]): int(val[1]) for val in list(zip(node_gdf['nodenumber'].tolist(), node_gdf['layer2ID'].tolist()))}
    # node_info['aquifer'].loc[node_info['layer']==2] = [lay2aq[int(node)] for node in node_info['nodenumber'].loc[node_info['layer']==2].tolist()]
    # for lay in range(nlay):
    #     h = 0

    del node_info['geometry']
    node_info.to_csv(os.path.join('input_data','rf.node.info'),index=False,sep='\t')

def make_2d_shp():
    # model_ws = os.path.join('model_ws','gridgen_test')
    # gridgen_ws = os.path.join(model_ws,'gridgen_ws')
    gridgen_ws = os.path.join('model_ws','RH_gridgen_ws')


    proj4 = '+proj=aea +lat_1=27.5 +lat_2=35 +lat_0=31.25 +lon_0=-100 +x_0=1500000 +y_0=6000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs'
    gridgen_ws = os.path.join('model_ws','RH_gridgen_ws')

    qtgrid = gpd.read_file(os.path.join(gridgen_ws,'qtgrid.shp'))
    qtgrid['layer'] = qtgrid['layer']+1
    qtgrid.crs = proj4
    usg_shp = qtgrid.copy()

    node_props = pd.read_csv(os.path.join('input_data','rf.node.info'),delim_whitespace=True)
    node_props_dict = node_props.set_index('nodenumber').to_dict()
    # node_props = node_props.loc[node_props['layer'] == 11].reset_index(drop=True) # get only bottom layer to get entire footprint
    node_props.drop_duplicates(subset=['x','y'],inplace=True)#.reset_index(drop=True)

    node_props['geometry'] = node_props.apply(lambda i: Point(i['x'], i['y']), axis=1)

    node_props_gdf = gpd.GeoDataFrame(node_props,crs=usg_shp.crs)
    node_props_gdf['CellID'] = np.arange(0,len(node_props_gdf)) + 1
    cellids = node_props_gdf['CellID'].unique()

    node_props_gdf = gpd.sjoin(usg_shp[['nodenumber','layer','top','bottom','geometry']],node_props_gdf[['CellID','x','y','area','geometry']],how='inner')

    # node_props_gdf.to_file(os.path.join('GIS','input_shapefiles','EL.Grid.2D.shp'))


    nlay = 25
    cols = []
    for lay in range(nlay):
        cols.append(f'nodeL{lay+1}')
    for lay in range(nlay):
        cols.append(f'topL{lay+1}')
    for lay in range(nlay):
        cols.append(f'botL{lay+1}')
    cols.append('geometry')
    data = {col:[] for col in cols}
    for cellid in cellids:
        tempdf = node_props_gdf.loc[node_props_gdf['CellID'] == cellid]
        geoms = tempdf['geometry'].tolist()
        geom = list(geoms)[0]
        for lay in range(nlay):
            if lay+1 not in tempdf['layer'].unique():
                node, top, bot = 0, 9999, 9999

            else:
                node = int(tempdf['nodenumber'].loc[tempdf['layer'] == lay+1].mean())
                top = tempdf['top'].loc[tempdf['layer'] == lay + 1].mean()
                bot = tempdf['bottom'].loc[tempdf['layer'] == lay+1].mean()

            data[f'nodeL{lay+1}'].append(node)
            data[f'topL{lay+1}'].append(top)
            data[f'botL{lay+1}'].append(bot)
        data['geometry'].append(geom)

    usg_shp = gpd.GeoDataFrame(data,crs=usg_shp.crs)

    usg_shp.sort_values(by=[f'nodeL{nlay-lay}' for lay in range(nlay)],inplace=True)
    usg_shp.to_file(os.path.join('GIS','input_shapefiles','RF_MODFLOW_USG_Grid_2d.shp'))

    print('end')
    # for lay in range(nlay):

def gen_oldnode2newnode():

    usg_shp = gpd.read_file(os.path.join('GIS','input_shapefiles','MODFLOW_USG_Grid_1d.shp'))
    usg_shp['oldlayer'] = usg_shp['layer']
    usg_shp['oldnodenumber'] = usg_shp['nodenumber']

    proj4 = '+proj=aea +lat_1=27.5 +lat_2=35 +lat_0=31.25 +lon_0=-100 +x_0=1500000 +y_0=6000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs'

    node_info = pd.read_csv(os.path.join('input_data','rf.node.info'),delim_whitespace=True)
    node_info['geometry'] = [Point(pts) for pts in list(zip(node_info['x'].astype(float).tolist(),node_info['y'].astype(float).tolist()))]
    node_gdf = gpd.GeoDataFrame(node_info,crs=proj4)

    nlay = 10
    node_gdf = gpd.sjoin(node_gdf[['nodenumber','layer','geometry']],usg_shp[['oldnodenumber','oldlayer','geometry']])
    node_gdf = node_gdf.loc[node_gdf['layer'] == node_gdf['oldlayer']]
    node_gdf.sort_values(['oldnodenumber','nodenumber'],inplace=True)

    node_gdf['OriginalNode'] = node_gdf['oldnodenumber']
    node_gdf['NewNode'] = node_gdf['nodenumber']
    node_gdf['Layer'] = node_gdf['layer']

    node_gdf = node_gdf[['Layer','OriginalNode','NewNode']]
    node_gdf.to_csv(os.path.join('input_data','rf.node.lookup.csv'),index=False)

def gen_oldnode2newnode2(proj4):

    usg_shp = gpd.read_file(os.path.join('GIS','input_shapefiles','grid_feet_projected.shp'))
    usg_shp['oldlayer'] = usg_shp['layer']
    usg_shp['oldnodenumber'] = usg_shp['nodenumber']

    # proj4 = '+proj=aea +lat_1=27.5 +lat_2=35 +lat_0=31.25 +lon_0=-100 +x_0=1500000 +y_0=6000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs'

    # node_info = pd.read_csv(os.path.join('input_data','rf.node.info'),delim_whitespace=True)
    node_info = gpd.read_file(os.path.join('model_ws','RH_gridgen_ws','qtgrid.shp'))
    node_info['layer'] = node_info['layer']+1
    node_info['geometry'] = node_info['geometry'].centroid

    node_gdf = gpd.GeoDataFrame(node_info,crs=proj4)

    node_gdf = gpd.sjoin(node_gdf[['nodenumber','layer','geometry']],usg_shp[['oldnodenumber','oldlayer','geometry']])
    node_gdf = node_gdf.loc[node_gdf['layer'] == node_gdf['oldlayer']]
    node_gdf.sort_values(['oldnodenumber','nodenumber'],inplace=True)

    node_gdf['OriginalNode'] = node_gdf['oldnodenumber']
    node_gdf['NewNode'] = node_gdf['nodenumber']
    node_gdf['Layer'] = node_gdf['layer']

    node_gdf = node_gdf[['Layer','OriginalNode','NewNode']]
    node_gdf.to_csv(os.path.join('input_data','rf.node.lookup.csv'),index=False)

def get_cl12(model_ws,disu):
    """
    Get the cl12 array
    Returns
    -------
    cl12 : ndarray
        A 1D vector of the cell connection distances, which are from the
        center of cell n to its shared face will cell m
    """
    iac = disu.iac.array
    njag = iac.sum()
    cl12 = np.empty((njag), dtype=np.float32)
    fname = os.path.join(model_ws, "qtg.c1.dat")
    f = open(fname, "r")
    cl12 = read1d(f, cl12)
    f.close()
    return cl12

def fix_dis():
    ogmodel_ws = os.path.join('model_ws','base.wel.fix.1929-2010.pst18')
    mfog = flopy.mfusg.MfUsg.load('gma12.nam',model_ws=ogmodel_ws,load_only=['disu'],forgive=True,check=False)

    ogdisu = mfog.disu
    fahl = ogdisu.fahl.array
    cl12 = ogdisu.cl12.array

    np.savetxt(os.path.join('model_ws','base.wel.fix.1929-2010.pst18','og.fahl.ref'),fahl)
    np.savetxt(os.path.join('model_ws','base.wel.fix.1929-2010.pst18','og.cl12.ref'),cl12)

    model_ws = os.path.join('model_ws','rf.base.wel.fix.1929-2010.pst18')
    mf = flopy.mfusg.MfUsg.load('rf.RH.nam',model_ws=model_ws,load_only=['disu'],forgive=True,check=False)

    disu = mf.disu
    fahl = disu.fahl.array
    cl12 = disu.cl12.array

    np.savetxt(os.path.join(model_ws,'fahl.ref'),fahl)
    np.savetxt(os.path.join(model_ws,'cl12.ref'),cl12)

    get_cl12(model_ws=os.path.join('model_ws','gma12_gridgen_ws'),disu=disu)

    print('eh')

def make_1d_shp():
    usg_shp = gpd.read_file(os.path.join('GIS','input_shapefiles','RF2_Oahu_MODFLOW_USG_Grid_2d.shp'))
    nlay = 25
    nodes, top, bot,geom = [],[],[],[]
    layer = []
    for lay in range(nlay):
        nodes += usg_shp[f'nodeL{lay+1}'].tolist()
        top += (usg_shp[f'topL{lay+1}']*3.28084).tolist()
        bot += (usg_shp[f'botL{lay+1}']*3.28084).tolist()
        layer += [lay+1] * len(usg_shp)
        geom += usg_shp['geometry'].tolist()

    data = {'nodenumber':nodes,
            'top':top,
            'bottom':bot,
            'layer':layer,
            'geometry':geom}

    usg_shp=gpd.GeoDataFrame(data,crs=usg_shp.crs)
    usg_shp = usg_shp.loc[usg_shp['nodenumber']>0]
    usg_shp.to_file(os.path.join('GIS','input_shapefiles','RF_OAHU_USG_Grid_1d.shp'))

def get_new2old(grid_df):
    df = grid_df[grid_df['NewNode'] > 0]

    # df.drop_duplicates(subset=['NewNodeShort'],inplace=True)
    df.sort_values(inplace=True,by='NewNode')
    df.set_index('NewNode',inplace=True)
    df_dict = df.to_dict()
    new2old = {node:df_dict['OriginalNode'][node] for node in df.index}
    # new2old = {}

    old2new = {node:[] for node in df['OriginalNode']}
    for newnode in new2old.keys():
        oldnode = new2old[newnode]
        old2new[oldnode].append(newnode)

    return new2old, old2new

def make_gsf_gnc():
    import pyemu

    gridgen_ws = os.path.join('model_ws', 'RH_gridgen_ws')
    shutil.copy(os.path.join('bin','3Dgnc.pl'),os.path.join(gridgen_ws,'3Dgnc.pl'))
    shutil.copy(os.path.join('bin','qt2gsf.in'),os.path.join(gridgen_ws,'qt2gsf.in'))
    shutil.copy(os.path.join('bin','makegnc.sh'),os.path.join(gridgen_ws,'makegnc.sh'))

    shutil.copy(os.path.join('bin','gwutils','gridgen2gsf.exe'),os.path.join(gridgen_ws,'gridgen2gsf.exe'))

    nodes_ndt = open(os.path.join(gridgen_ws,'nodes - Copy.ndt'), 'r')
    nodes_ndt_lines = nodes_ndt.readlines()
    nodes_ndt.close()

    nodes_ndt_lay = os.path.join(gridgen_ws,'nodes.ndt')

    f = open(nodes_ndt_lay, 'w')

    for number, line  in enumerate(nodes_ndt_lines):
        l = nodes_ndt_lines[1].split(' ')[-1][0]
        if number == 0 or line.split(' ')[-1] == '1\n':
            f.write(line)
        else:
            continue
    f.close()
    print('Wrote node.ndt')

    qtgrid = open(os.path.join(gridgen_ws, 'quadtreegrid - Copy.tsf'), 'r')
    qtgrid_lines = qtgrid.readlines()
    qtgrid.close()

    qtgrid_lay = os.path.join(gridgen_ws, 'quadtreegrid.tsf')

    f = open(qtgrid_lay, 'w')

    for number, line in enumerate(qtgrid_lines):
        l = qtgrid_lines[1].split(' ')[1][1:3]
        if number == 0:
            f.write('12444\n')
        elif line.split(' ')[1][1:3] == '1,':
            f.write(line)
        else:
            continue
    f.close()
    print('Wrote quadtreegrid.tsf')


    pyemu.utils.run('gridgen2gsf < qt2gsf.in',cwd=gridgen_ws)

    # import subprocess
    # p = subprocess.Popen("C:\Program Files\Git\git-bash.exe ./makegnc.sh",
    # bufsize = -1,
    # executable = None,
    # stdin = None,
    # stdout = None,
    # stderr = None,
    # preexec_fn = None,
    # close_fds = False,
    # shell = False,
    # cwd = gridgen_ws,
    # )

def make_aux_info_for_pumping():

    grid_df = pd.read_csv(os.path.join('input_data', 'rf.node.lookup.csv'))
    new2old, old2new = get_new2old(grid_df)

    oldgam_ref_df = pd.read_csv(os.path.join('input_data','gamlay.ref'),names=['nodenumber','oldlay'],delim_whitespace=True)
    fout = open(os.path.join('input_data','rf.gamlay.ref'),'w')
    for i, dfrow in oldgam_ref_df.iterrows():
        oldnode = dfrow['nodenumber']
        layer = dfrow['oldlay']
        for node in old2new[oldnode]:
            row = [node,layer]
            fout.write('{}\n'.format(' '.join([str(x) for x in row])))
    fout.close()

    nlay = 25
    usg_shp = gpd.read_file(os.path.join('GIS','input_shapefiles','RF_MODFLOW_USG_Grid_2d.shp'))
    usg_shp.sort_values(by=[f'nodeL{10-lay}' for lay in range(nlay)],inplace=True)
    data = []
    for lay in range(nlay):
        for i, dfrow in usg_shp.iterrows():
            node = dfrow[f'nodeL{lay+1}']
            if node > 0:
                row = [node, lay+1]
                for l in np.arange(lay+1, nlay):
                    node2 = dfrow[f'nodeL{l+1}']
                    if node2 > 0:
                        row.append(node2)
                data.append(row)

    fout = open(os.path.join('input_data','rf.grid.nl'),'w')
    for row in data:
        fout.write('{}\n'.format(' '.join([str(x) for x in row])))

    fout.close()
    print('rf.grid.nl')


def main():
    case = '02_restart_500'
    model_ws = os.path.join('model_ws', f'{case}')
    make_gsf_gnc()

    # gen_usg_model(plot=False)
    # gen_new_node_info()
    # make_gsf_gnc()
    # make_2d_shp()
    # make_1d_shp()



    # get_OG_grid_info()
    # # get_refined_cells()
    # save_layer_footprint()
    # # create_coarseGrid(nrow, ncol, nlay, delr, delc, xoff, yoff, rotation, proj4)
    # update_disu_props(test=False)

    # make_1d_shp()
    # get_tops(nrow=39, ncol=41, plot=True)





if __name__ == '__main__':

    main()