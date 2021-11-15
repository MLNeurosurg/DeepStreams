# DeepStreams
Code repository for the following manuscript: *Spatiotemporal analysis of glioma heterogeneity reveals Col1A1 as an actionable 1 target to disrupt tumor mesenchymal differentiation, invasion and malignancy.*
[[paper]](https://www.biorxiv.org/content/10.1101/2020.12.01.404970v2)

Repo to train and validate U-Net model for semantic segmentation of oncostreams in H&E images. Installation and demo takes less than a minute. 

# Installation

1. Install miniconda: follow instructions
    [here](https://docs.conda.io/en/latest/miniconda.html)
2. Create conda environment:  
    ```console
    conda create -n deepstreams python=3.8
    ```
3. Activate conda environment:  
    ```console
    conda activate deepstreams
    ```
4. Install package and dependencies  
    ```console
    <cd /path/to/repo/dir>
    pip install -e .
    ```
4. Train model
    ```console
    python main.py
    ```

```bibtex
@ARTICLE{Comba2021-dr,
  title    = "Spatiotemporal analysis of glioma heterogeneity reveals {Col1A1}
              as an actionable 1 target to disrupt tumor mesenchymal
              differentiation, invasion and malignancy",
  author   = "Comba, Andrea and Motsch, Sebastien and Dunn, Patrick J and
              Hollon, Todd C and Argento, Anna E and Zamler, Daniel B and Kish,
              Phillips E and Kahana, Alon and Kleer,Celina G and Castro, Maria
              Graciela and Lowenstein, Pedro R",
  abstract = "Intra-tumoral heterogeneity and diffuse infiltratjkion are
              hallmarks of glioblastoma that challenge 18 treatment efficacy.
              However, the mechanisms that set up both tumor heterogeneity and
              invasion 19 remain poorly understood. Herein, we present a
              comprehensive spatiotemporal study that aligns 20 distinctive
              intra-tumoral histopathological structures, oncostreams, with
              dynamic properties and 21 a unique, actionable, spatial
              transcriptomic signature. Oncostreams are dynamic multicellular
              22 fascicles of spindle-like and aligned cells with mesenchymal
              properties. Their density correlates 23 with tumor aggressiveness
              in genetically engineered mouse glioma models, and high grade 24
              human gliomas. Oncostreams facilitate the intra-tumoral
              distribution of tumoral and non-25 tumoral cells, and the
              invasion of the normal brain. These fascicles are defined by a
              specific 26 molecular signature that regulates their organization
              and function. Oncostreams structure and 27 function depend on
              overexpression of COL1A1. COL1A1 is a central gene in the dynamic
              28 organization of glioma mesenchymal transformation, and a
              powerful regulator of glioma 29 malignant behavior. Inhibition of
              COL1A1 eliminated oncostreams, reprogramed the malignant 30
              histopathological phenotype, reduced expression of the
              mesenchymal associated genes, and 31 prolonged animal survival.
              Oncostreams represent a novel pathological marker of potential
              value 32 for diagnosis, prognosis, and treatment. \#\#\#
              Competing Interest Statement The authors have declared no
              competing interest.",
  journal  = "bioRxiv",
  pages    = "2020.12.01.404970",
  month    =  may,
  year     =  2021,
  language = "en"
}


```
