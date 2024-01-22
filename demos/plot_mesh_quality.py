import numpy as np
import matplotlib.pyplot as plt
import sys

case = 4 # 1: Camembert, 2: Generic polygon, 3: SEAM, 4: Marmousi 

mq_path = "/home/santos/spyro/paper_moving_mesh/mesh_quality/"

if case==1: #{{{

    SAVE = 0
    AMR = 0
    QUAD = 0
        
    if QUAD==0:
        title = "Triangle"
    elif QUAD==1:
        title = "Quadrilateral"
    else:
        sys.exit("tile not defined")

    if AMR==0:
        MFUNC = 0
        mq_file = mq_path + "camembert_model" + "_QUAD=" + str(QUAD) + "_AMR=" + str(AMR) + "_M=" + str(MFUNC) + ".npy"
        with open(mq_file, 'rb') as f:
            mesh_quality = np.load(f)
        
        print(title)
        print("min mesh quality (non-adapted) = " +  str(round(min(mesh_quality),2)))
        print("mean mesh quality (non-adapted) = " + str(round(np.mean(mesh_quality),2)))
            
    else:

        MFUNC = 2 # paper M1=M2, and M2=M1
        mq_file_M1 = mq_path + "camembert_model" + "_QUAD=" + str(QUAD) + "_AMR=" + str(AMR) + "_M=" + str(MFUNC) + ".npy"
        with open(mq_file_M1, 'rb') as f:
            mesh_quality_M1 = np.load(f)
        
        MFUNC = 1 # paper M1=M2, and M2=M1
        mq_file_M2 = mq_path + "camembert_model" + "_QUAD=" + str(QUAD) + "_AMR=" + str(AMR) + "_M=" + str(MFUNC) + ".npy"
        with open(mq_file_M2, 'rb') as f:
            mesh_quality_M2 = np.load(f)
        
        MFUNC = 3 
        mq_file_M3 = mq_path + "camembert_model" + "_QUAD=" + str(QUAD) + "_AMR=" + str(AMR) + "_M=" + str(MFUNC) + ".npy"
        with open(mq_file_M3, 'rb') as f:
            mesh_quality_M3 = np.load(f)


        plt.style.use('tableau-colorblind10')
        
        n_bins = 20 
        plt.hist(mesh_quality_M1, n_bins, alpha=1.0, edgecolor='black', label="M1", histtype="stepfilled", density=True)
        plt.hist(mesh_quality_M2, n_bins, alpha=0.7, edgecolor='black', label="M2", histtype="stepfilled", density=True)
        plt.hist(mesh_quality_M3, n_bins, alpha=0.6, edgecolor='black', label="M3", histtype="stepfilled", density=True)

        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Element quality',size=14)
        plt.xticks(fontsize=14)
        plt.xlim(0.29, 1.01)
        plt.ylabel('% Elements',size=14)
        plt.yticks(fontsize=14)
        plt.title(title,size=18)
        plt.legend(loc='upper left',fontsize=14)
        plt.tight_layout()
       
        if SAVE:
            plt.savefig(mq_path+"camembert_"+title+".png",dpi=400)
        else:
            plt.show()

        print(title)
        print("min mesh quality (M1) = " +  str(round(min(mesh_quality_M1),2)))
        print("min mesh quality (M2) = " +  str(round(min(mesh_quality_M2),2)))
        print("min mesh quality (M3) = " +  str(round(min(mesh_quality_M3),2)))
        print("mean mesh quality (M1) = " + str(round(np.mean(mesh_quality_M1),2)))
        print("mean mesh quality (M2) = " + str(round(np.mean(mesh_quality_M2),2)))
        print("mean mesh quality (M3) = " + str(round(np.mean(mesh_quality_M3),2)))


#}}}
if case==2: #{{{

    SAVE = 0 
    AMR = 0
    QUAD = 1
    
    if QUAD==0:
        title = "Triangle"
    elif QUAD==1:
        title = "Quadrilateral"
    else:
        sys.exit("tile not defined")

    if AMR==0:
        MFUNC = 0
        mq_file = mq_path + "generic_polygon_model" + "_QUAD=" + str(QUAD) + "_AMR=" + str(AMR) + "_M=" + str(MFUNC) + ".npy"
        with open(mq_file, 'rb') as f:
            mesh_quality = np.load(f)
        
        print(title)
        print("min mesh quality (non-adapted) = " +  str(round(min(mesh_quality),2)))
        print("mean mesh quality (non-adapted) = " + str(round(np.mean(mesh_quality),2)))
            
    else:
    
        MFUNC = 2 # paper M1=M2, and M2=M1
        mq_file_M1 = mq_path + "generic_polygon_model" + "_QUAD=" + str(QUAD) + "_AMR=" + str(AMR) + "_M=" + str(MFUNC) + ".npy"
        with open(mq_file_M1, 'rb') as f:
            mesh_quality_M1 = np.load(f)
        
        MFUNC = 1 # paper M1=M2, and M2=M1
        mq_file_M2 = mq_path + "generic_polygon_model" + "_QUAD=" + str(QUAD) + "_AMR=" + str(AMR) + "_M=" + str(MFUNC) + ".npy"
        with open(mq_file_M2, 'rb') as f:
            mesh_quality_M2 = np.load(f)
        
        MFUNC = 3 
        mq_file_M3 = mq_path + "generic_polygon_model" + "_QUAD=" + str(QUAD) + "_AMR=" + str(AMR) + "_M=" + str(MFUNC) + ".npy"
        with open(mq_file_M3, 'rb') as f:
            mesh_quality_M3 = np.load(f)


        plt.style.use('tableau-colorblind10')
        
        n_bins = 20 
        plt.hist(mesh_quality_M1, n_bins, alpha=1.0, edgecolor='black', label="M1", histtype="stepfilled", density=True)
        plt.hist(mesh_quality_M2, n_bins, alpha=0.7, edgecolor='black', label="M2", histtype="stepfilled", density=True)
        plt.hist(mesh_quality_M3, n_bins, alpha=0.6, edgecolor='black', label="M3", histtype="stepfilled", density=True)

        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Element quality',size=14)
        plt.xticks(fontsize=14)
        plt.xlim(0.29, 1.01)
        plt.ylabel('% Elements',size=14)
        plt.yticks(fontsize=14)
        plt.title(title,size=18)
        plt.legend(loc='upper left',fontsize=14)
        plt.tight_layout()
        
        if SAVE:
            plt.savefig(mq_path+"generic_polygon_"+title+".png",dpi=400)
        else:
            plt.show()

        print(title)
        print("min mesh quality (M1) = " +  str(round(min(mesh_quality_M1),2)))
        print("min mesh quality (M2) = " +  str(round(min(mesh_quality_M2),2)))
        print("min mesh quality (M3) = " +  str(round(min(mesh_quality_M3),2)))
        print("mean mesh quality (M1) = " + str(round(np.mean(mesh_quality_M1),2)))
        print("mean mesh quality (M2) = " + str(round(np.mean(mesh_quality_M2),2)))
        print("mean mesh quality (M3) = " + str(round(np.mean(mesh_quality_M3),2)))

#}}}
if case==3: #{{{

    SAVE = 0
    AMR = 0
    
    QUAD = 1  # fixed
    MFUNC = 3  
   
    if AMR==0:
        MFUNC = 0

    mq_file = mq_path + "seam" + "_QUAD=" + str(QUAD) + "_AMR=" + str(AMR) + "_M=" + str(MFUNC) + ".npy"
    with open(mq_file, 'rb') as f:
        mesh_quality = np.load(f)
    
    if QUAD==0:
        sys.exit("tile not defined")
    elif QUAD==1:
        title = "Quadrilateral"
    else:
        sys.exit("tile not defined")

    plt.style.use('tableau-colorblind10')
    
    n_bins = 20 
    plt.hist(mesh_quality, n_bins, alpha=1.0, edgecolor='black', label="M3", histtype="stepfilled", density=True)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Element quality',size=14)
    plt.xticks(fontsize=14)
    plt.xlim(0.29, 1.01)
    plt.ylabel('% Elements',size=14)
    plt.yticks(fontsize=14)
    plt.title(title,size=18)
    plt.legend(loc='upper left',fontsize=14)
    plt.tight_layout()
    
    if SAVE:
        plt.savefig(mq_path+"seam_"+title+".png",dpi=400)
    else:
        plt.show()

    print(title)
    print("min mesh quality (M3) = " +  str(round(min(mesh_quality),2)))
    print("mean mesh quality (M3) = " + str(round(np.mean(mesh_quality),2)))

#}}}
if case==4: #{{{

    SAVE = 0
    AMR = 1
    
    QUAD = 0 # fixed
    MFUNC = 3
   
    if AMR==0:
        MFUNC = 0

    mq_file = mq_path + "marmousi" + "_QUAD=" + str(QUAD) + "_AMR=" + str(AMR) + "_M=" + str(MFUNC) + ".npy"
    with open(mq_file, 'rb') as f:
        mesh_quality = np.load(f)
    
    if QUAD==0:
        title = "Triangle (unstructured mesh)"
    elif QUAD==1:
        sys.exit("tile not defined")
    else:
        sys.exit("tile not defined")

    plt.style.use('tableau-colorblind10')
    
    n_bins = 20 
    plt.hist(mesh_quality, n_bins, alpha=1.0, edgecolor='black', label="M3", histtype="stepfilled", density=True)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Element quality',size=14)
    plt.xticks(fontsize=14)
    plt.xlim(0.29, 1.01)
    plt.ylabel('% Elements',size=14)
    plt.yticks(fontsize=14)
    plt.title(title,size=18)
    plt.legend(loc='upper left',fontsize=14)
    plt.tight_layout()
    
    if SAVE:
        plt.savefig(mq_path+"marmousi_"+title+".png",dpi=400)
    else:
        plt.show()

    print(title)
    print("min mesh quality (M3) = " +  str(round(min(mesh_quality),2)))
    print("mean mesh quality (M3) = " + str(round(np.mean(mesh_quality),2)))

#}}}
