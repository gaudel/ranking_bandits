<!-- Our title -->
<div align="center">
  <h3>ranking_bandits </h3>
</div>

<!-- Short description -->
<p align="center">
A Python-toolkit for bandit algorithms choosing a ranked-list of $K$ items among $L$ at each iteration
</p>



<!-- Draw horizontal rule -->
<hr>


> This contains the code used for the research articles
> * [Parametric Graph for Unimodal Ranking Bandit]() (Camille-Sovaneary Gauthier, Romaric Gaudel, Elisa Fromont, and Boammani Aser Lompo,  2021) presented at the 38$^{th}$ International Conference on Machine Learning ([ICML'21](https://icml.cc/Conferences/2021)).
> * [Bandit Algorithm for Both Unknown BestPosition and Best Item Display on Web Pages]() (Camille-Sovaneary Gauthier, Romaric Gaudel, and Elisa Fromont, 2021) presented at the 19$^{th}$ Symposium on Intelligent Data Analysis ([IDA'21](https://ida2021.org/)).


## Requirements

* Python >= 3.6
* libraries (see `requirements.txt` build with Python 3.8.7)
    * numpy
    * scipy
    * (for PMED) tensorflow 
    * (for `exp.py` and `run_experiments.py`) docopt
    * (for `run_experiments.py`) matplotlib

## (Re)run experiments

See corresponding README file:

* [README_IDA_2021.md](README_IDA_2021.md)
* [README_ICML_2021.md](README_ICML_2021.md)


## Referencing ranking_bandits

If you use `ranking_bandits` in a scientific publication, we would appreciate citations:

```bibtex
@InProceedings{pmlr-v139-gauthier21a,
  title = 	 {Parametric Graph for Unimodal Ranking Bandit},
  author =       {Gauthier, Camille-Sovanneary and Gaudel, Romaric and Fromont, Elisa and Lompo, Boammani Aser},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {3630--3639},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/gauthier21a/gauthier21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/gauthier21a.html}
}
```


#### Acknowledgments
Authors would like to thank Louis Vuitton for granting the PhD  at the root of this library.
