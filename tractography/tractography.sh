#!/bin/bash

# Execute with e.g.: bash tractography.sh all ../../HCP_diffusion_MV/data/ 128

# some colors for fancy logging
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

subject_id=$1  # e.g., 100206 or 'all' to process all subjects
data_dir=$2    # folder that contains the subjects
num_jobs=$3    # number of parallel jobs

threading="-nthreads 1" # Set the max number of threads process can use (e.g. number of cores)

process_subject() {
    local subject_id=$1
    local data_dir=$2

    start_time=$(date +%s)  # Start time

    dmri_dir="${data_dir}/${subject_id}/dMRI"
    anat_dir="${data_dir}/${subject_id}/anat"

    # Create a output directory, skip patient if already created
    output_dir="${data_dir}/${subject_id}/output"
    if [ ! -d "${output_dir}" ]; then
        mkdir -p "${output_dir}"
    else
        echo -e "${RED}[INFO]${NC} `date`: Skipping ${subject_id}, output directory already exists."
        return
    fi

    log_file="${data_dir}/${subject_id}/output/tractography_log.txt"

    echo -e "${GREEN}[INFO]${NC} `date`: Starting tractography for: ${subject_id}" | tee -a "${log_file}"

    # First convert the initial diffusion image to .mif (~10sec)
    dwi_mif="${dmri_dir}/dwi.mif"
    if [ ! -f ${dwi_mif} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Converting dwi image to mif" | tee -a "${log_file}"
        # check for eddy rotated bvecs
        mrconvert "${dmri_dir}/data.nii.gz" "${dwi_mif}" \
                -fslgrad "${dmri_dir}/bvecs" "${dmri_dir}/bvals" \
                -datatype float32 -strides 0,0,0,1 ${threading} -info 2>&1 | tee -a "${log_file}"
    fi

    # Then, extract mean B0 image (~1sec)
    dwi_meanbzero="${dmri_dir}/dwi_meanbzero.mif"
    dwi_meanbzero_nii="${dmri_dir}/dwi_meanbzero.nii.gz"
    if [ ! -f ${dwi_meanbzero} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Extracting mean B0 image" | tee -a "${log_file}"

        # extract mean b0
        dwiextract ${threading} -info "${dwi_mif}" -bzero - | mrmath ${threading} -info - mean -axis 3 "${dwi_meanbzero}" 2>&1
        mrconvert "${dwi_meanbzero}" "${dwi_meanbzero_nii}" ${threading} -info 2>&1 | tee -a "${log_file}"
    fi

    # Then, create a dwi brain mask (the provided bedpostX mask is not that accurate) (~2sec)
    dwi_meanbzero_brain="${dmri_dir}/dwi_meanbzero_brain.nii.gz"
    dwi_meanbzero_brain_mask="${dmri_dir}/dwi_meanbzero_brain_mask.nii.gz"
    if [ ! -f ${dwi_meanbzero_brain_mask} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Computing dwi brain mask" | tee -a "${log_file}"

        # Approach 2: using FSL BET (check https://github.com/sina-mansour/UKB-connectomics/commit/463b6553b5acd63f14a45ef7120145998e0a5139)

        # skull stripping to get a mask
        bet "${dwi_meanbzero_nii}" "${dwi_meanbzero_brain}" -m -R -f 0.2 -g -0.05 2>&1 | tee -a "${log_file}"
    fi

    #################################################################
    ############# CONSTRAINED SPHERICAL DECONVOLUTION ###############
    #################################################################
    start_time_csp=$(date +%s)

    # Estimate the response function using the dhollander method (~4min)
    wm_txt="${dmri_dir}/wm.txt"
    gm_txt="${dmri_dir}/gm.txt"
    csf_txt="${dmri_dir}/csf.txt"
    if [ ! -f ${wm_txt} ] || [ ! -f ${gm_txt} ] || [ ! -f ${csf_txt} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Estimation of response function using dhollander" | tee -a "${log_file}"
        dwi2response dhollander "${dwi_mif}" "${wm_txt}" "${gm_txt}" "${csf_txt}" \
                                -voxels "${dmri_dir}/voxels.mif" ${threading} -info 2>&1 | tee -a "${log_file}"
    fi

    # Multi-Shell, Multi-Tissue Constrained Spherical Deconvolution (~33min)
    wm_fod="${dmri_dir}/wmfod.mif"
    gm_fod="${dmri_dir}/gmfod.mif"
    csf_fod="${dmri_dir}/csffod.mif"
    dwi_mask_dilated="${dmri_dir}/dwi_meanbzero_brain_mask_dilated_2.nii.gz"
    if [ ! -f ${wm_fod} ] || [ ! -f ${gm_fod} ] || [ ! -f ${csf_fod} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Running Multi-Shell, Multi-Tissue Constrained Spherical Deconvolution" | tee -a "${log_file}"
        
        # First, creating a dilated brain mask (https://github.com/sina-mansour/UKB-connectomics/issues/4)
        maskfilter -npass 2 "${dwi_meanbzero_brain_mask}" dilate "${dwi_mask_dilated}" ${threading} -info 2>&1 | tee -a "${log_file}"

        # Now, perfoming CSD with the dilated mask
        dwi2fod msmt_csd "${dwi_mif}" -mask "${dwi_mask_dilated}" "${wm_txt}" "${wm_fod}" \
                "${gm_txt}" "${gm_fod}" "${csf_txt}" "${csf_fod}" ${threading} -info 2>&1 | tee -a "${log_file}"
    fi

    # mtnormalise to perform multi-tissue log-domain intensity normalisation (~5sec)
    wm_fod_norm="${dmri_dir}/wmfod_norm.mif"
    gm_fod_norm="${dmri_dir}/gmfod_norm.mif"
    csf_fod_norm="${dmri_dir}/csffod_norm.mif"
    if [ ! -f ${wm_fod_norm} ] || [ ! -f ${gm_fod_norm} ] || [ ! -f ${csf_fod_norm} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Running multi-tissue log-domain intensity normalisation" | tee -a "${log_file}"
        
        # First, creating an eroded brain mask (https://github.com/sina-mansour/UKB-connectomics/issues/5)
        maskfilter -npass 2 "${dwi_meanbzero_brain_mask}" erode "${dmri_dir}/dwi_meanbzero_brain_mask_eroded_2.nii.gz" ${threading} -info 2>&1 | tee -a "${log_file}"

        # Now, perfoming mtnormalise
        mtnormalise "${wm_fod}" "${wm_fod_norm}" "${gm_fod}" "${gm_fod_norm}" "${csf_fod}" \
                    "${csf_fod_norm}" -mask "${dmri_dir}/dwi_meanbzero_brain_mask_eroded_2.nii.gz" ${threading} -info 2>&1 | tee -a "${log_file}"
    fi

    # create a combined fod image for visualization
    vf_mif="${dmri_dir}/vf.mif"
    if [ ! -f ${vf_mif} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Generating a visualization file from normalized FODs" | tee -a "${log_file}"
        mrconvert ${threading} -info -coord 3 0 "${wm_fod_norm}" - | mrcat "${csf_fod_norm}" "${gm_fod_norm}" - "${vf_mif}" 2>&1 | tee -a "${log_file}"
    fi


    end_time_csp=$(date +%s)  # End time
    elapsed_time_csp=$((end_time_csp - start_time_csp))
    

    #################################################################
    ################### CREATING TISSUE BOUNDARY ####################
    #################################################################
    start_time_tb=$(date +%s)


    # Create a mask of white matter gray matter interface using 5 tissue type segmentation (~70sec)
    T1_brain_nii="${anat_dir}/T1w_acpc_dc_restore_brain.nii.gz"
    T1_brain="${anat_dir}/T1w_acpc_dc_restore_brain.mif"
    if [ ! -f ${T1_brain} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Converting T1 brain image to mif" | tee -a "${log_file}"
        mrconvert "${T1_brain_nii}" "${T1_brain}" 2>&1 | tee -a "${log_file}"
    fi

    T1_nii="${anat_dir}/T1w_acpc_dc_restore.nii.gz"
    T1="${anat_dir}/T1w_acpc_dc_restore.mif"
    if [ ! -f ${T1} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Converting T1 image to mif" | tee -a "${log_file}"
        mrconvert "${T1_nii}" "${T1}" 2>&1 | tee -a "${log_file}"
    fi

    parcellation_nii="${anat_dir}/aparc+aseg.nii.gz"
    parcellation="${anat_dir}/aparc+aseg.mif"
    if [ ! -f ${parcellation} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Converting parcellation image to mif" | tee -a "${log_file}"
        mrconvert "${parcellation_nii}" "${parcellation}" 2>&1 | tee -a "${log_file}"
    fi

    seg_5tt_T1="${dmri_dir}/seg_5tt_T1.mif"
    seg_5tt="${dmri_dir}/seg_5tt.mif"

    T1_brain_dwi="${dmri_dir}/T1_brain_dwi.mif"
    T1_brain_mask="${anat_dir}/T1_brain_mask.nii.gz"

    gmwm_seed_T1="${dmri_dir}/gmwm_seed_T1.mif"
    gmwm_seed="${dmri_dir}/gmwm_seed.mif"
    transform_DWI_T1_FSL="${dmri_dir}/diff2struct_fsl.txt"
    transform_DWI_T1="${dmri_dir}/diff2struct_mrtrix.txt"

    if [ ! -f ${gmwm_seed} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Running 5ttgen to get gray matter white matter interface mask" | tee -a "${log_file}"
        # First create the 5tt image
        # Option 1: generate the 5TT image based on a FreeSurfer parcellation image
        # 5ttgen freesurfer "${parcellation}" "${seg_5tt_T1}" -nocrop ${threading} -info 2>&1 | tee -a "${log_file}"
        
        # Option 2: generate the 5TT image based on a T1 image with FSL
        5ttgen fsl ${threading} "${T1_brain}" "${seg_5tt_T1}" -premasked -info 2>&1 | tee -a "${log_file}"
        # 5ttgen fsl "${T1_brain_nii}" "${seg_5tt_T1}" -premasked ${threading} -info
        # 5ttgen fsl "${T1_nii}" "${seg_5tt_T1}" ${threading} -nocrop -info  
        # 5ttgen fsl "${T1}" "${seg_5tt_T1}" ${threading} -nocrop ${threading} -info  

        # Next generate the boundary ribbon
        5tt2gmwmi ${threading} "${seg_5tt_T1}" "${gmwm_seed_T1}" -info 2>&1 | tee -a "${log_file}"

        # Coregistering the Diffusion and Anatomical Images
        # Perform rigid body registration
        flirt -in "${dwi_meanbzero_brain}" -ref "${T1_brain_nii}" \
            -cost normmi -dof 6 -omat "${transform_DWI_T1_FSL}" 2>&1 | tee -a "${log_file}"
        transformconvert "${transform_DWI_T1_FSL}" "${dwi_meanbzero_brain}" \
                        "${T1_brain}" flirt_import "${transform_DWI_T1}" 2>&1 | tee -a "${log_file}"

        # Perform transformation of the boundary ribbon from T1 to DWI space
        mrtransform "${seg_5tt_T1}" "${seg_5tt}" -linear "${transform_DWI_T1}" -inverse ${threading} -info 2>&1 | tee -a "${log_file}"
        mrtransform "${T1_brain}" "${T1_brain_dwi}" -linear "${transform_DWI_T1}" -inverse ${threading} -info 2>&1 | tee -a "${log_file}"
        mrtransform "${gmwm_seed_T1}" "${gmwm_seed}" -linear "${transform_DWI_T1}" -inverse ${threading} -info 2>&1 | tee -a "${log_file}"
        
        # Visualize result
        # mrview T1_brain_dwi.mif -overlay.load gmwm_seed.mif -overlay.colourmap 2 -overlay.load gmwm_seed_T1.mif -overlay.colourmap 1
        # mrview ../anat/T1_brain.mif -overlay.load gmwm_seed.mif -overlay.colourmap 2 -overlay.load gmwm_seed_T1.mif -overlay.colourmap 1
        # mrview dwi_meanbzero.mif -overlay.load gmwm_seed.mif -overlay.colourmap 2 -overlay.load gmwm_seed_T1.mif -overlay.colourmap 1
        # mrview dwi_meanbzero.mif -overlay.load seg_5tt.mif -overlay.colourmap 2 -overlay.load seg_5tt_T1.mif -overlay.colourmap 1
    fi

    end_time_tb=$(date +%s)  # End time
    elapsed_time_tb=$((end_time_tb - start_time_tb))



    #################################################################
    ########################## STREAMLINES ##########################
    #################################################################
    start_time_tractography=$(date +%s)
    streamlines=100K
    # Create streamlines
    tracts="${dmri_dir}/tracts_${streamlines}.tck"
    tractstats="${dmri_dir}/stats/${subject_id}_tracts_${streamlines}_stats.json"
    mkdir -p "${dmri_dir}/stats"
    if [ ! -f ${tracts} ]; then
        echo -e "${GREEN}[INFO]${NC} `date`: Running probabilistic tractography" | tee -a "${log_file}"
        tckgen -seed_gmwmi "${gmwm_seed}" -act "${seg_5tt}" -select "${streamlines}" \
                            -maxlength 250 -cutoff 0.1 ${threading} "${wm_fod_norm}" "${tracts}" -power 0.5 \
                            -info -samples 3  2>&1 | tee -a "${log_file}" #-output_stats "${tractstats}"
        # Visualize result
        # tckedit "${tracts}" -number 200k "${dmri_dir}/tracts_200k.tck"
        # mrview dwi_meanbzero.mif -tractography.load smallertracts_200k.tck
        # mrview anat/T1w_acpc_dc_restore_brain.nii.gz -tractography.load dMRI/tracts_200k.tck
        # mrview 103818/anat/T1w_acpc_dc_restore_brain.nii.gz -tractography.load 103818/dMRI/tracts_200k.tck
    fi

    tracts_MNI="${dmri_dir}/tracts_${streamlines}_MNI.tck"
    if [ ! -f ${tracts_MNI} ]; then
        # Prepare deformation field file
        mrconvert ${anat_dir}/standard2acpc_dc.nii.gz ${dmri_dir}/tmp-[].nii -force
        mv ${dmri_dir}/tmp-0.nii ${dmri_dir}/x.nii
        mrcalc ${dmri_dir}/x.nii -neg ${dmri_dir}/tmp-0.nii -force
        warpconvert ${dmri_dir}/tmp-[].nii displacement2deformation ${dmri_dir}/acpc2MNI_mrtrix.nii.gz -force
        rm ${dmri_dir}/x.nii ${dmri_dir}/tmp-?.nii

        # Transform tracts to MNI space
        tcktransform "${tracts}" \
                    "${dmri_dir}/acpc2MNI_mrtrix.nii.gz" \
                    "${tracts_MNI}"
        # Generate subsample for visualization
        # tckedit "${tracts_MNI}" -number 200k "${dmri_dir}/tracts_200k_MNI.tck"
        # mrview anat/T1w_restore_brain.nii.gz -tractography.load dMRI/tracts_200k_MNI.tck
    fi

    end_time_tractography=$(date +%s)  # End time
    elapsed_time_tractography=$((end_time_tractography - start_time_tractography))

    #################################################################
    ################## MAP STRUCTURAL CONNECTIVITY ##################
    #################################################################

    # Write tracts to vtk file
    tracts_vtk=${output_dir}/streamlines_${streamlines}.vtk
    if [ ! -f ${tracts_vtk} ]; then
        tckconvert -binary ${tracts} ${tracts_vtk} 2>&1 | tee -a "${log_file}" # might give error as -binary became an option in rtrix 3.0.4 (download with conda install -c mrtrix3 mrtrix3)
    fi
    tracts_vtk_MNI=${output_dir}/streamlines_${streamlines}_MNI.vtk
    if [ ! -f ${tracts_vtk_MNI} ]; then
        tckconvert -binary ${tracts_MNI} ${tracts_vtk_MNI} 2>&1 | tee -a "${log_file}" 
    fi

    # Define both parcellations
    parcellations=("aparc+aseg" "aparc.a2009s+aseg")

    # Loop over both parcellations
    for parc in "${parcellations[@]}"; do

        # Define paths for the current parcellation
        parcellation_nii="${anat_dir}/${parc}.nii.gz"
        parcellation="${anat_dir}/${parc}.mif"
        parcellation_converted="${anat_dir}/${parc}_mrtrix.mif"
        connectome_matrix="${output_dir}/connectome_matrix_${parc}.csv"

        # Convert parcellation image to mif if not already converted
        if [ ! -f ${parcellation} ]; then
            echo -e "${GREEN}[INFO]${NC} `date`: Converting ${parc} parcellation image to mif" | tee -a "${log_file}"
            mrconvert "${parcellation_nii}" "${parcellation}" 2>&1 | tee -a "${log_file}"
        fi

        # Convert Freesurfer labels to MRtrix if not already converted
        if [ ! -f ${parcellation_converted} ]; then
            echo -e "${GREEN}[INFO]${NC} `date`: Converting Freesurfer labels for ${parc} to MRtrix" | tee -a "${log_file}"
            # labelconvert "${parcellation}" $FREESURFER_HOME/FreeSurferColorLUT.txt $MRTRIX3_HOME/share/mrtrix3/labelconvert/fs_default.txt "${parcellation_converted}" 2>&1 | tee -a "${log_file}"
            labelconvert "${parcellation}" ./txt_files/FreeSurferColorLUT.txt ./txt_files/fs_${parc}.txt "${parcellation_converted}" 2>&1 | tee -a "${log_file}"
        fi

        # Generate connectome matrix if not already generated
        if [ ! -f ${connectome_matrix} ]; then
            echo -e "${GREEN}[INFO]${NC} `date`: Computing connectome matrix from streamline count for ${parc}" | tee -a "${log_file}"

            # Record the start time
            start_time=$(date +%s)

            tck2connectome ${threading} -info -symmetric \
                            "${tracts}" "${parcellation_converted}" "${connectome_matrix}" \
                            -out_assignments ${output_dir}/labels_${streamlines}_${parc}.txt 2>&1 | tee -a "${log_file}"

            # Record the end time
            end_time=$(date +%s)

            # Calculate the elapsed time and log it
            elapsed_time=$((end_time - start_time))
            echo -e "${GREEN}[INFO]${NC} `date`: tck2connectome completed for ${parc} in ${elapsed_time} seconds" | tee -a "${log_file}"

            # Generate the connectome matrix plot
            python plot_connectome.py "${connectome_matrix}" "${output_dir}/connectome_matrix_${streamlines}_${parc}.png" "Connectome matrix subject ${subject_id} (${parc})" 2>&1 | tee -a "${log_file}"
        fi

    done

    echo -e "${GREEN}[INFO]${NC} `date`: Finished tractography for: ${subject_id}" | tee -a "${log_file}"
    
    end_time=$(date +%s)  # End time
    elapsed_time=$((end_time - start_time))

    echo ""
    echo -e "${GREEN}[INFO]${NC} `date`: Constrained Spherical Deconvolution took: ${elapsed_time_csp} seconds." | tee -a "${log_file}"
    echo -e "${GREEN}[INFO]${NC} `date`: Generating tissue boundary took: ${elapsed_time_tb} seconds." | tee -a "${log_file}"
    echo -e "${GREEN}[INFO]${NC} `date`: Tractography took: ${elapsed_time_tractography} seconds." | tee -a "${log_file}"
    echo -e "${GREEN}[INFO]${NC} `date`: Finished processing ${subject_id}. Total time: ${elapsed_time} seconds." | tee -a "${log_file}"

    # Call the Python script to clean the log file
    python3 ./clean_log.py "${log_file}"
    echo "Log file cleaned!"
}

# if [ "$subject_id" = "all" ]; then
#     for subject_dir in "${data_dir}"/*/ ; do
#     # for subject_dir in $(ls -d "${data_dir}"/*/ | tac); do
#         real_subject_id=$(basename "${subject_dir}")
#         (
#             process_subject "${real_subject_id}" "${streamlines}" "${data_dir}"
#         ) 
#     done
# else
#     process_subject "${subject_id}" "${streamlines}" "${data_dir}"
# fi


export -f process_subject  # Export the function for parallel

if [ "${subject_id}" == "all" ]; then
    # Get a list of all subjects in the data directory
    subjects=$(ls "${data_dir}" | grep -E '^[0-9]{6}$')
else
    subjects="${subject_id}"
fi

# Run the process_subject function in parallel for each subject
echo "${subjects}" | parallel -j "${num_jobs}" process_subject {} "${data_dir}"