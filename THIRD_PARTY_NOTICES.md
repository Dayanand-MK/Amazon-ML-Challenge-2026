# Third-party notices

The project LICENSE applies to original Hacksmiths code and the locally trained final model artifact. It does not relicense challenge datasets, the supplied official validator/template/rules, historical code owned by others, or third-party dependencies. The model uses no pretrained weights.

LightGBM 4.6.0 is MIT-licensed: https://github.com/microsoft/LightGBM/blob/v4.6.0/LICENSE . Its copyright belongs to Microsoft Corporation and the LightGBM developers. Its exact installed license, including the upstream copyright notice, is reproduced in licenses/LightGBM_LICENSE.txt.

Other dependencies retain their own licenses: scikit-learn, NumPy, pandas and SciPy (BSD variants), RapidFuzz (MIT), text-unidecode (Artistic License as declared by its package metadata). text-unidecode provides generic local character transliteration, not business identity data. The nonselected logistic regression baseline uses scikit-learn. The final estimator uses LightGBM.

utils/validate_submission.py, reports/official_challenge_rules.md and reports/official_documentation_template.md were copied from the organizers' student_resource folder supplied by the user. The validator is unmodified; reports/submission_validation.json records its SHA-256. These challenge materials retain the organizers' rights.
