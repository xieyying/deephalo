# streamlit_app.py
import streamlit as st

# 创建标题
st.title('Bioinformatics and Chemoinformatics')

# 创建副标题
st.header('Biosynthetic Gene Cluster Identification')

# 创建列表
st.markdown("""
- AntiSMASH : Based on protein similarity mining bacteria, fungi, plant secondary metabolite biosynthetic gene clusters
- PRISM4 : Bacterial natural product biosynthetic gene cluster, structure and activity prediction
- EvoMining : Based on evolution mining bacterial and archaeal natural product biosynthetic gene clusters
- CO-OCCUR : Based on evolutionary mining fungal natural product biosynthetic gene clusters
- RRE-finder : Mining new RiPP gene clusters by identifying RRE domains
- DecRippter : Based on pan-genome and machine learning to mine new RiPP gene clusters
- ARTS 2.0 : Based on antibiotic resistance gene targeting to mine natural product biosynthetic gene clusters
""")

# 创建副标题
st.header('Biosynthetic Gene Cluster Comparison')

# 创建列表
st.markdown("""
- BiSCAPE/COROSON : Similarity comparison, clustering and diversity analysis of gene clusters from different genomes
- BiG-SLICE : Similarity comparison and clustering analysis of millions of BGCs
""")

# 创建副标题
st.header('Biosynthetic Gene Cluster Database')

# 创建列表
st.markdown("""
- MiBIG : Experimentally validated biosynthetic gene cluster database
- AntiSMASH-DB V3 : High-quality natural product biosynthetic gene cluster database derived from bacteria and a small amount of fungi and archaea predicted by antiSMASH V5.2
- IMG-ABC V5 : Natural product biosynthetic gene cluster database predicted by antiSMASH V5 and a small amount of experimentally verified
- Prospect : Fungal natural product biosynthetic gene cluster database predicted by antiSMASH V4
- BIG-FAM : Natural product biosynthetic gene cluster family database predicted by antiSMASH from bacteria, archaea, fungi and metagenomes
""")

# 创建副标题
st.header('Chemoinformatics')

# 创建副标题
st.header('Mass Spectrometry-Based Natural Product Analysis')

# 创建列表
st.markdown("""
- MZmine 3.0 : Mass spectrometry data processing software
- GNPS : Comprehensive platform for mass spectrometry data clustering, deduplication and annotation analysis and storage and sharing
- Moldiscovery : Natural product deduplication and automatic annotation based on theoretical mass spectrometry library search
- CycloNovo : Cyclopeptide de novo analysis
- NRPro : Peptide natural product deduplication and automatic annotation based on theoretical mass spectrometry library search
- SIRIUS : De novo annotation of molecular formula and molecular structure based on high-resolution mass spectrometry
""")

# 创建副标题
st.header('NMR-Based Natural Product Analysis')

# 创建列表
st.markdown("""
- SMART 2.1 : 1H-13C HSQC spectrum automatic analysis platform
- DP4-AI : 13C and 1H NMR data automatic processing and annotation program
- NP-MRD : Natural product nuclear magnetic resonance database
""")

# 创建副标题
st.header('Microbial Natural Product Database')

# 创建列表
st.markdown("""
- NP Atlas : Microbial natural product database
- Streptome-DB 3.0 : Streptomyces natural product database
- NORINE : Non-ribosomal peptide compound database
- COCONUT : Open source natural product database collection
""")