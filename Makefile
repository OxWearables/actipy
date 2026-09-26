PRE2026_PINS := \
	"numpy<=2.3.5" \
	"scipy<=1.16.3" \
	"pandas<=2.3.3" \
	"statsmodels<=0.14.6" \
	"tqdm<=4.67.1" \
	"ruff<=0.14.10" \
	"autopep8<=2.3.2" \
	"ipython<=9.8.0" \
	"ipdb<=0.13.13" \
	"twine<=6.2.0" \
	"tomli<=2.3.0" \
	"pytest<=9.0.2" \
	"joblib<=1.5.3" \
	"memray<=1.19.1" \
	"sphinx<=9.1.0" \
	"sphinx_rtd_theme<=3.0.2" \
	"readthedocs-sphinx-search<=0.3.2" \
	"docutils<=0.17.1"

.PHONY: install install-pre2026

install:
	pip install -e .[dev,docs]

install-pre2026:
	pip install -e .[dev,docs] $(PRE2026_PINS)
