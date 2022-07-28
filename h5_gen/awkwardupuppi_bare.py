import h5py
import sys
import awkward as ak
import numpy as np
import uproot as uproot
import smear_chargedhadrons as sch
from awkward import Array
from awkward.layout import ListOffsetArray64

#np.random.seed(0)
file = uproot.open(sys.argv[1])

events = file['Delphes']
nEvents = 100

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
gen_pfs_genvtx = read_branch('PileUpMix.GenVtxIdx') + 1
gen_pfs_z_vtx = gen_vtx_z[gen_pfs_genvtx]

gen_pfs = {'pt':gen_pfs_pt, 'eta':gen_pfs_eta, 'phi':gen_pfs_phi, 'e':gen_pfs_e, 'pid':gen_pfs_pid, 'z':gen_pfs_z, 'charge':gen_pfs_charge, 'genvtx':gen_pfs_genvtx, 'z_vtx':gen_pfs_z_vtx}
gen_vtx = {'x':gen_vtx_x, 'y':gen_vtx_y, 'z':gen_vtx_z}

# min pt > 0.5, np.abs(eta) < 5
measurable_mask = (gen_pfs_pt > 0.5) & (np.abs(gen_pfs_eta) < 5)

# neutrino mask, true when the particle is a neutrino
neutrino_mask = (np.abs(gen_pfs_pid) == 12) | (np.abs(gen_pfs_pid) == 14) | (np.abs(gen_pfs_pid) == 16)

for keys in gen_pfs.keys():
    gen_pfs[keys] = gen_pfs[keys][measurable_mask & ~neutrino_mask]

# not smearing energy for now
"""
# Smearing photon energies
def smear_em(gen_pfs_pt,gen_pfs_eta,gen_pfs_e,gen_pfs_id,pid):
    from awkward import Array
    from awkward.layout import ListOffsetArray64
    
    #Convert it to a 1D numpy array and perform smearing
    numpy_e_arr = np.asarray(gen_pfs_e.layout.content)
    numpy_pt_arr = np.asarray(gen_pfs_pt.layout.content)
    numpy_eta_arr = np.asarray(gen_pfs_eta.layout.content)
    numpy_id_arr = np.asarray(gen_pfs_id.layout.content)

    if pid == 22:
        smeared_e_arr = np.random.normal(numpy_e_arr, numpy_e_arr*0.01)
    elif pid == 11:
        smeared_e_arr = np.where(np.abs(numpy_eta_arr)<1.5,
                                 np.random.normal(numpy_e_arr, numpy_e_arr*0.028),
                                 numpy_e_arr)
        smeared_e_arr = np.where(((np.abs(numpy_eta_arr)>1.5) & (np.abs(numpy_eta_arr)<=1.75)),
                                 np.random.normal(numpy_e_arr, numpy_e_arr*0.037),
                                 smeared_e_arr)
        smeared_e_arr = np.where(((np.abs(numpy_eta_arr)>1.75) & (np.abs(numpy_eta_arr)<=2.15)),
                                 np.random.normal(numpy_e_arr, numpy_e_arr*0.038),
                                 smeared_e_arr)
        smeared_e_arr = np.where(((np.abs(numpy_eta_arr)>2.15) & (np.abs(numpy_eta_arr)<=3.00)),
                                 np.random.normal(numpy_e_arr, numpy_e_arr*0.044),
                                 smeared_e_arr)
        smeared_e_arr = np.where(((np.abs(numpy_eta_arr)>3.00) & (np.abs(numpy_eta_arr)<=4.00)),
                                 np.random.normal(numpy_e_arr, numpy_e_arr*0.10),
                                 smeared_e_arr)

    # only apply smearing to desired particles matching pid
    smeared_e_arr = np.where(np.abs(numpy_id_arr)==pid,smeared_e_arr,numpy_e_arr)
    

    smeared_pt_arr = np.where(np.abs(numpy_id_arr)==pid,smeared_e_arr/np.cosh(numpy_eta_arr),numpy_pt_arr)
    #smeared_pt_arr = numpy_pt_arr

    #Convert it back to awkward form
    return Array(ListOffsetArray64(gen_pfs_e.layout.offsets, Array(smeared_e_arr).layout)),Array(ListOffsetArray64(gen_pfs_pt.layout.offsets, Array(smeared_pt_arr).layout))


gen_pfs_e, gen_pfs_pt = smear_em(gen_pfs_pt,gen_pfs_eta,gen_pfs_e,gen_pfs_pid,22)
gen_pfs_e, gen_pfs_pt = smear_em(gen_pfs_pt,gen_pfs_eta,gen_pfs_e,gen_pfs_pid,11)
gen_pfs_e, gen_pfs_pt = sch.smear_chargedhad(gen_pfs_pt,gen_pfs_eta,gen_pfs_e,gen_pfs_pid,gen_pfs_charge,22)


#gen_pfs_e, gen_pfs_pt = sch.smear_hcal(gen_pfs_pt,gen_pfs_eta,gen_pfs_e,gen_pfs_pid,gen_pfs_charge,22)



def process_data(pfs_data):
    '''
    Processes data to as measured reasonably by detector
    1. converts electrons outside of the barrel to photons
    ... and other stuff
    '''
    # Implemented later, using bare data for now
    raise(NotImplementedError('use bare data for now'))

def one_hot_pid(gen_pfs_pid,gen_pfs_charge,gen_pfs_eta):
    
    #Convert it to a 1D numpy array
    numpy_pid_arr = np.asarray(gen_pfs_pid.layout.content)
    numpy_charge_arr = np.asarray(gen_pfs_charge.layout.content)
    numpy_eta_arr = np.asarray(gen_pfs_eta.layout.content)

    # Convert electrons outside tracker to photons
    new_pid_arr = np.where(((np.abs(numpy_eta_arr)>3.0) & (np.abs(numpy_pid_arr)==11)),22.,numpy_pid_arr)
    # Convert charged hadrons outside tracker to neutral hadrons (0)
    new_pid_arr = np.where(((np.abs(numpy_eta_arr)>3.0) & (numpy_charge_arr!=0) & (np.abs(numpy_pid_arr)>13)),0.,new_pid_arr)
    # Convert all neutral hadrons to default 0
    new_pid_arr = np.where(((numpy_charge_arr==0) & ( np.abs(numpy_pid_arr)>22)),0.,new_pid_arr)
    # Convert all charged hadrons within tracker to pion or non-pion
    new_pid_arr = np.where(((np.abs(numpy_eta_arr)<3.0) & (numpy_charge_arr!=0) & (np.abs(numpy_pid_arr)!=211)),1.,new_pid_arr)
    new_pid_arr = np.where(((np.abs(numpy_eta_arr)<3.0) & (numpy_charge_arr!=0) & (np.abs(numpy_pid_arr)==211)),2.,new_pid_arr)
    # Give electrons inside tracker to 3
    new_pid_arr = np.where(((np.abs(numpy_eta_arr)<3.0) & (np.abs(numpy_pid_arr)==11)),3.,new_pid_arr)
    # Give photons 4
    new_pid_arr = np.where((np.abs(new_pid_arr)==22),4.,new_pid_arr)

    #print(np.any(new_pid_arr == 3))

    return Array(ListOffsetArray64(gen_pfs_pid.layout.offsets, Array(new_pid_arr).layout))



def disable_charge(gen_pfs_charge,gen_pfs_eta,gen_pfs_genvtx):
    #Convert it to a 1D numpy array
    numpy_charge_arr = np.asarray(gen_pfs_charge.layout.content)
    numpy_eta_arr = np.asarray(gen_pfs_eta.layout.content)
    numpy_genvtx_arr = np.asarray(gen_pfs_genvtx.layout.content)

    new_charge_arr = np.where(np.abs(numpy_eta_arr)<3.0,numpy_charge_arr,0.)
    new_vtx_arr = np.where(numpy_charge_arr==0,-1,numpy_genvtx_arr)
    return Array(ListOffsetArray64(gen_pfs_charge.layout.offsets, Array(new_charge_arr).layout)),  Array(ListOffsetArray64(gen_pfs_genvtx.layout.offsets, Array(new_vtx_arr).layout))


# gen_pfs_charge, gen_pfs_vtx = disable_charge(gen_pfs_charge,gen_pfs_eta,gen_pfs_genvtx)

# kinematics mask
#kinematics_mask = (gen_pfs_pt>1.0) & (np.abs(gen_pfs_eta)<5.)

"""

def simplify_pid(gen_pfs_pid):
    
    #Convert it to a 1D numpy array
    np_pid = np.asarray(gen_pfs_pid.layout.content)
    np_pid = np.abs(np_pid)
    # replace pid {11, 13, 22, 130, 211, 321, 2112, 2212} with {0, 1, 2, 3, 4, 5, 6, 7}
    np_pid = np.where(np_pid==11,0,np_pid)
    np_pid = np.where(np_pid==13,1,np_pid)
    np_pid = np.where(np_pid==22,2,np_pid)
    np_pid = np.where(np_pid==130,3,np_pid)
    np_pid = np.where(np_pid==211,4,np_pid)
    np_pid = np.where(np_pid==321,5,np_pid)
    np_pid = np.where(np_pid==2112,6,np_pid)
    np_pid = np.where(np_pid==2212,7,np_pid)
    return Array(ListOffsetArray64(gen_pfs_pid.layout.offsets, Array(np_pid).layout))

gen_pfs['pid'] = simplify_pid(gen_pfs['pid'])


gen_vtx['id'] = ak.local_index(gen_vtx['z'])
# gives the shape of awkward array to index through

def fill_ndf_and_vtxpt(gen_pfs_genvtx, gen_pfs_pt, gen_vtx_id, gen_pfs_charge):
    # in ideal case, filter neutrals, but not now
    # gen_pfs_vtx = gen_pfs_genvtx[gen_pfs_charge != 0]

    big_vtxcounts = []
    big_vtxpts = []
    # event loop
    for e in range(len(gen_pfs_genvtx)):
        vtxcounts = []
        vtxpts = []
        
        #Convert it to a 1D numpy array
        numpy_vtx_arr = np.asarray(gen_pfs_genvtx[e])
        numpy_pt = np.asarray(gen_pfs_pt[e])

        # gen vtx loop
        for v in gen_vtx_id[e]:
            vtxcounts.append((numpy_vtx_arr == v).sum())
            vtxpts.append(numpy_pt[numpy_vtx_arr == v].sum())


        big_vtxcounts.append(vtxcounts)
        big_vtxpts.append(vtxpts)
    
    
    return ak.Array(big_vtxcounts), ak.Array(big_vtxpts)


gen_vtx['ndf'], gen_vtx['total_pt'] = fill_ndf_and_vtxpt(gen_pfs['genvtx'], gen_pfs['pt'], gen_vtx['id'], gen_pfs['charge'])

def rearrange_vertex(gen_vtx, gen_pfs_genvtx):
    '''
    Rearranges the vertices according to the total_pt and updates the truth value of particles accordingly
    '''
    new_vtx_x = []
    new_vtx_y = []
    new_vtx_z = []
    new_vtx_ndf = []
    new_vtx_total_pt = []
    new_gen_pfs_genvtx = []
    for e in range(len(gen_pfs_genvtx)):
        event_vtx_pt = gen_vtx['total_pt'][e]
        event_vtx_ndf = gen_vtx['ndf'][e]
        sorted_indices = np.lexsort((-event_vtx_ndf, -event_vtx_pt))
        event_vtx_ndf = event_vtx_ndf[sorted_indices]
        # remove the vertices that have no particles
        active_vtx_mask = (event_vtx_ndf > 0)
        new_vtx_x.append(gen_vtx['x'][e][sorted_indices][active_vtx_mask])
        new_vtx_y.append(gen_vtx['y'][e][sorted_indices][active_vtx_mask])
        new_vtx_z.append(gen_vtx['z'][e][sorted_indices][active_vtx_mask])
        new_vtx_ndf.append(gen_vtx['ndf'][e][sorted_indices][active_vtx_mask])
        new_vtx_total_pt.append(gen_vtx['total_pt'][e][sorted_indices][active_vtx_mask])


        truth_indices = np.argsort(sorted_indices)
        new_event_gen_pfs_genvtx = []
        for t in gen_pfs_genvtx[e]:
            new_event_gen_pfs_genvtx.append(truth_indices[t])
        new_gen_pfs_genvtx.append(new_event_gen_pfs_genvtx)

    new_gen_vtx = {'x': new_vtx_x, 'y': new_vtx_y, 'z': new_vtx_z, 'ndf': new_vtx_ndf, 'total_pt': new_vtx_total_pt, 'id':ak.local_index(new_vtx_z)}
    return new_gen_vtx, new_gen_pfs_genvtx

gen_vtx, gen_pfs['genvtx'] = rearrange_vertex(gen_vtx, gen_pfs['genvtx'])




def get_reco_z(gen_pfs_z,gen_pfs_charge):
    from awkward import Array
    from awkward.layout import ListOffsetArray64
    
    #Convert it to a 1D numpy array
    numpy_charge_arr = np.asarray(gen_pfs_charge.layout.content)
    numpy_z_arr = np.asarray(gen_pfs_z.layout.content)

    new_z_arr = np.where(numpy_charge_arr!=0.0,numpy_z_arr,-199.)
    # clamp between -200 and 200
    new_z_arr = np.clip(new_z_arr,-200.,200.)
    return Array(ListOffsetArray64(gen_pfs_charge.layout.offsets, Array(new_z_arr).layout))


gen_pfs['reco_z_vtx'] = get_reco_z(gen_pfs['z_vtx'], gen_pfs['charge'])
gen_pfs['reco_z'] = get_reco_z(gen_pfs['z'], gen_pfs['charge'])

final_pfs = {}
for key in gen_pfs.keys():
    if key != 'genvtx':
        final_pfs[key] = np.array(ak.fill_none(ak.pad_none(gen_pfs[key],7000,clip=True),0,axis=-1))
final_pfs['genvtx'] = np.array(ak.fill_none(ak.pad_none(gen_pfs['genvtx'],7000,clip=True),-99,axis=-1))
# stack all the features from dictonary into a single array
final_pfs_features = np.stack((final_pfs['pt'], final_pfs['eta'], final_pfs['phi'], final_pfs['e'], final_pfs['pid'], final_pfs['reco_z'], final_pfs['charge'], final_pfs['reco_z_vtx']), axis=-1)


final_vtx = {}
for key in gen_vtx.keys():
    final_vtx[key] = np.array(ak.fill_none(ak.pad_none(gen_vtx[key],200,clip=True),0,axis=-1))

final_vtx_features = np.stack((final_vtx['x'], final_vtx['y'], final_vtx['z'], final_vtx['ndf'], final_vtx['total_pt']), axis=-1)

hf = h5py.File(sys.argv[2], 'w')

hf.create_dataset('pfs', data=final_pfs_features)
hf.create_dataset('pfs_shape', data=final_pfs_features.shape)
hf.create_dataset('z', data=final_pfs['z'])
hf.create_dataset('z_shape', data=final_pfs['z'].shape)
hf.create_dataset('vtx', data=final_vtx_features)
hf.create_dataset('vtx_shape', data=final_vtx_features.shape)
hf.create_dataset('truth', data=final_pfs['genvtx'])
hf.create_dataset('truth_shape', data=final_pfs['genvtx'].shape)
hf.create_dataset('vtx_truthidx', data=final_vtx['id'])
hf.create_dataset('vtx_truthidx_shape', data=final_vtx['id'].shape)
hf.create_dataset('n', data=final_pfs['pt'].shape[0])

hf.close()

