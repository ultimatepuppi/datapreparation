import h5py
import sys
import awkward as ak
import numpy as np
import uproot as uproot
from awkward import Array
from awkward.layout import ListOffsetArray64

#np.random.seed(0)

file = uproot.open(sys.argv[1])

events = file['Delphes']
nEvents = 1000

# read branches
def read_branch(branchname):
    return events[branchname].array(entry_stop=nEvents)

gen_vtx = read_branch('GenVertex_size')
gen_vtx_x = read_branch('GenVertex.X')
gen_vtx_y = read_branch('GenVertex.Y')
gen_vtx_z = read_branch('GenVertex.Z')
gen_pfs_pt = read_branch('PileUpMix.PT')
gen_pfs_eta = read_branch('PileUpMix.Eta')
gen_pfs_phi = read_branch('PileUpMix.Phi')
gen_pfs_e = read_branch('PileUpMix.E')
gen_pfs_pid = read_branch('PileUpMix.PID')
gen_pfs_z = read_branch('PileUpMix.Z')
gen_pfs_charge = read_branch('PileUpMix.Charge')
gen_pfs_genvtx = read_branch('PileUpMix.GenVtxIdx')
gen_pfs_genvtx = gen_pfs_genvtx + 1

correct_zs = []

gen_pfs_z_new = gen_vtx_z[gen_pfs_genvtx]
gen_pfs_z = gen_pfs_z_new


# neutrino mask
neutrino_mask = (np.abs(gen_pfs_pid) != 12) & (np.abs(gen_pfs_pid) != 14) & (np.abs(gen_pfs_pid) != 16)

def filter_neutrinos(arr):
    newarr = arr[neutrino_mask]
    return newarr

gen_pfs_pt = filter_neutrinos(gen_pfs_pt)
gen_pfs_eta = filter_neutrinos(gen_pfs_eta)
gen_pfs_phi = filter_neutrinos(gen_pfs_phi)
gen_pfs_e = filter_neutrinos(gen_pfs_e)
gen_pfs_pid = filter_neutrinos(gen_pfs_pid)
gen_pfs_charge = filter_neutrinos(gen_pfs_charge)
gen_pfs_z = filter_neutrinos(gen_pfs_z)
gen_pfs_genvtx = filter_neutrinos(gen_pfs_genvtx)

gen_vtx_id = ak.local_index(gen_vtx_z)


def fill_ndf(gen_pfs_vtx,gen_pfs_charge,gen_vtx_id):
    from awkward import Array
    from awkward.layout import ListOffsetArray64

    # filter neutrals
    gen_pfs_vtx = gen_pfs_vtx[gen_pfs_charge != 0]

    big_vtxcounts = []
    # event loop
    for e in range(len(gen_pfs_vtx)):
        vtxcounts = []
        
        #Convert it to a 1D numpy array
        numpy_vtx_arr = np.asarray(gen_pfs_vtx[e])

        # gen vtx loop
        for v in gen_vtx_id[e]:
            vtxcounts.append((numpy_vtx_arr == v).sum())

        big_vtxcounts.append(vtxcounts)
    #print(big_vtxcounts)
    
    
    return ak.Array(big_vtxcounts)

gen_vtx_ndf = fill_ndf(gen_pfs_genvtx,gen_pfs_charge,gen_vtx_id)

def get_reco_z(gen_pfs_z,gen_pfs_charge):
    from awkward import Array
    from awkward.layout import ListOffsetArray64
    
    #Convert it to a 1D numpy array
    numpy_charge_arr = np.asarray(gen_pfs_charge.layout.content)
    numpy_z_arr = np.asarray(gen_pfs_z.layout.content)

    new_z_arr = np.where(numpy_charge_arr!=0.0,numpy_z_arr,-199.)
    return Array(ListOffsetArray64(gen_pfs_charge.layout.offsets, Array(new_z_arr).layout))


gen_pfs_recoz = get_reco_z(gen_pfs_z,gen_pfs_charge)

final_pfs_pt = np.array(ak.fill_none(ak.pad_none(gen_pfs_pt,7000,clip=True),0,axis=-1))
final_pfs_eta = np.array(ak.fill_none(ak.pad_none(gen_pfs_eta,7000,clip=True),0,axis=-1))
final_pfs_phi = np.array(ak.fill_none(ak.pad_none(gen_pfs_phi,7000,clip=True),0,axis=-1))
final_pfs_e = np.array(ak.fill_none(ak.pad_none(gen_pfs_e,7000,clip=True),0,axis=-1))
final_pfs_pid = np.array(ak.fill_none(ak.pad_none(gen_pfs_pid,7000,clip=True),0,axis=-1))
final_pfs_charge = np.array(ak.fill_none(ak.pad_none(gen_pfs_charge,7000,clip=True),0,axis=-1))
final_pfs_z = np.array(ak.fill_none(ak.pad_none(gen_pfs_z,7000,clip=True),0,axis=-1))
final_pfs_recoz = np.array(ak.fill_none(ak.pad_none(gen_pfs_recoz,7000,clip=True),0,axis=-1))
#final_pfs_vtx = np.array(ak.fill_none(ak.pad_none(gen_pfs_vtx,7000,clip=True),-99,axis=-1))
final_pfs_genvtx = np.array(ak.fill_none(ak.pad_none(gen_pfs_genvtx,7000,clip=True),-99,axis=-1))

final_pfs_features = np.stack((final_pfs_pt,final_pfs_eta,final_pfs_phi,final_pfs_e,final_pfs_pid,final_pfs_charge,final_pfs_recoz),axis=-1)

#print(final_pfs_vtx)
#print(final_pfs_vtx.shape)
#print(final_pfs_features)
#print(final_pfs_features.shape)


final_vtx_x = np.array(ak.fill_none(ak.pad_none(gen_vtx_x,200,clip=True),0,axis=-1))
final_vtx_y = np.array(ak.fill_none(ak.pad_none(gen_vtx_y,200,clip=True),0,axis=-1))
final_vtx_z = np.array(ak.fill_none(ak.pad_none(gen_vtx_z,200,clip=True),0,axis=-1))
final_vtx_ndf = np.array(ak.fill_none(ak.pad_none(gen_vtx_ndf,200,clip=True),0,axis=-1))
final_vtx_id = np.array(ak.fill_none(ak.pad_none(gen_vtx_id,200,clip=True),0,axis=-1))

final_vtx_features = np.stack((final_vtx_x,final_vtx_y,final_vtx_z,final_vtx_ndf),axis=-1)


hf = h5py.File(sys.argv[2], 'w')

hf.create_dataset('pfs', data=final_pfs_features)
hf.create_dataset('pfs_shape', data=final_pfs_features.shape)
hf.create_dataset('z', data=final_pfs_z)
hf.create_dataset('z_shape', data=final_pfs_z.shape)
#hf.create_dataset('edge_start', data=final_pfs_vtx)
#hf.create_dataset('edge_start_shape', data=final_pfs_vtx.shape)
#hf.create_dataset('edge_stop', data=final_vtx_id)
#hf.create_dataset('edge_stop_shape', data=final_vtx_id.shape)
hf.create_dataset('vtx', data=final_vtx_features)
hf.create_dataset('vtx_shape', data=final_vtx_features.shape)
hf.create_dataset('truth', data=final_pfs_genvtx)
hf.create_dataset('truth_shape', data=final_pfs_genvtx.shape)
hf.create_dataset('vtx_truthidx', data=final_vtx_id)
hf.create_dataset('vtx_truthidx_shape', data=final_vtx_id.shape)
hf.create_dataset('n', data=final_pfs_genvtx.shape[0])

hf.close()

