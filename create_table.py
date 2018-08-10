import pandas as pd
import os

RESULT_DIRECTORY = '.\\results'

# Baseline results
"""
bresults = pd.read_csv(os.path.join(RESULT_DIRECTORY, 'baseline_results.csv'))

results_formatted = pd.DataFrame()

for i in bresults.index:
	results_formatted.loc[i, "runID"] = bresults.loc[i, "22"]
	results_formatted.loc[i, "method"] = bresults.loc[i, "0"]
	results_formatted.loc[i, "rmse_train"] = bresults.loc[i, "12"]
	results_formatted.loc[i, "rmse_test"] = bresults.loc[i, "13"]
	
	for j in range(3, 11):
		col = bresults.loc[i, str(j)]
		results_formatted.loc[i, col.replace(' ', '')] = bresults.loc[i, str(j + 11)]
		
results_formatted['White_Hispanic'] = results_formatted.rmse_test_white - results_formatted.rmse_test_hispanic 
results = results_formatted.groupby('method')[['White_Hispanic']].mean()
"""

# Baseline results One Sample
def add_significance(series, method):

	v = series.loc[method]
	
	if hasattr(series, "test"):
		tv = series.test.loc[method]
		if (tv > 1.645):
			return " & \\textbf{%.3f}" %v
		else:
				return " & %.3f" %v
	else:
		return " & %.3f" %v


oresults = pd.read_csv(os.path.join(RESULT_DIRECTORY, 'baseline_results_altsplit_small.txt'), sep=',')
print(oresults.columns)	

for c in list(oresults.columns):
	oresults.rename(columns={c: c.replace(' ', '')}, inplace=True)

oresults['White Hispanic'] = (oresults.mse_test_white - oresults.mse_test_hispanic) / (oresults.var_test_white + oresults.var_test_hispanic) ** 0.5
oresults['White Black'] = (oresults.mse_test_white - oresults.mse_test_black) / (oresults.var_test_white + oresults.var_test_black) ** 0.5
oresults['White Asian'] = (oresults.mse_test_white - oresults.mse_test_asian) / (oresults.var_test_white + oresults.var_test_asian) ** 0.5
oresults['Female Male'] = (oresults.mse_test_f - oresults.mse_test_m) / (oresults.var_test_f + oresults.var_test_m) ** 0.5


baseline_export = oresults[['White Hispanic', 'White Black', 'White Asian', 'Female Male']]

baseline_export.loc['pmf', 'Method'] = 'PMF'
baseline_export.loc['pmf_bias', 'Method'] = 'PMFB'
baseline_export.loc['bpmf', 'Method'] = 'BPMF'
baseline_export.loc['fastfm', 'Method'] = 'FM'
baseline_export.loc['fastfm_features', 'Method'] = 'FMX'

baseline_export = baseline_export.set_index('Method')


#write latex tables

"""		
		
with open(os.path.join(RESULT_DIRECTORY, 'test_baseline_results.tex'), 'w') as output:
	output.write("\\" + "begin{table} \n")
	output.write("\\centering  \n")
	output.write("\\label{tab:2}  \n")
	output.write("\\begin{tabular}{c|ccc|c} \n")
	
	output.write('Recommender' + ''.join(' & %s' %c for c in list(baseline_export.columns)) +  '\\' + '\\' + '\n')
	output.write("\\hline \n")
	output.write("\\hline \n")
	
	for meth in baseline_export.index:
		output.write(meth + ''.join(add_significance(v) for v in baseline_export.loc[meth, :].values) +  '\\' + '\\' + '\n')
	
	output.write("\\end{tabular} \n")
	output.write("\\caption{t-stats of difference in mean squared errors. Values in bold characters show a statistically significant difference at the $5\%$ using a left tail t-distribution.}" + '\n')
	
	output.write("\\end{table} \n")
	
output.close
"""

oresults['All'] = oresults['mse_test']
oresults['White'] = oresults['mse_test_white']
oresults['Hispanic'] = oresults['mse_test_hispanic']
oresults['Black'] = oresults['mse_test_black']
oresults['Asian'] = oresults['mse_test_asian']
oresults['Female'] = oresults['mse_test_f']
oresults['Male'] = oresults['mse_test_m']


oresults.loc['pmf', 'Method'] = 'PMF'
oresults.loc['pmf_bias', 'Method'] = 'PMFB'
oresults.loc['bpmf', 'Method'] = 'BPMF'
oresults.loc['fastfm', 'Method'] = 'FM'
oresults.loc['fastfm_features', 'Method'] = 'FMX'

oresults = oresults.set_index('Method')

oresults['Hispanic'].test = abs((oresults.mse_test_white - oresults.mse_test_hispanic) / (oresults.var_test_white + oresults.var_test_hispanic) ** 0.5)
oresults['Black'].test = abs((oresults.mse_test_white - oresults.mse_test_black) / (oresults.var_test_white + oresults.var_test_black) ** 0.5)
oresults['Asian'].test = abs((oresults.mse_test_white - oresults.mse_test_asian) / (oresults.var_test_white + oresults.var_test_asian) ** 0.5)
oresults['Male'].test = abs((oresults.mse_test_f - oresults.mse_test_m) / (oresults.var_test_f + oresults.var_test_m) ** 0.5)

baseline_export2 = oresults[['All', 'White', 'Hispanic', 'Black', 'Asian', 'Female', 'Male']]
baseline_export2['Hispanic'].test = oresults['Hispanic'].test
baseline_export2['Black'].test = oresults['Black'].test 
baseline_export2['Asian'].test = oresults['Asian'].test 
baseline_export2['Male'].test = oresults['Male'].test

print(baseline_export2['Black'].test)

with open(os.path.join(RESULT_DIRECTORY, 'mse_baseline_results.tex'), 'w') as output:
	output.write("\\" + "begin{table} \n")
	output.write("\\centering  \n")
	output.write("\\label{tab:1}  \n")
	output.write("\\begin{tabular}{c|ccccc|cc} \n")
	
	output.write('Recommender' + ''.join(' & %s' %c for c in list(baseline_export2.columns)) +  '\\' + '\\' + '\n')
	output.write("\\hline \n")
	output.write("\\hline \n")
	
	for meth in baseline_export2.index:
		output.write(meth + ''.join(add_significance(baseline_export2[v], meth) for v in list(baseline_export2.columns)) +  '\\' + '\\' + '\n')
	
	output.write("\\end{tabular} \n")
	output.write("\\caption{Mean squared errors on the test sample across demographic groups. "
				"Values in bold characters show a statistically significant difference at the $95\%$ using a left tail t-distribution. "
				"Recommender systems are trained using only students with demographic information }" + '\n')
	
	output.write("\\end{table} \n")
	
output.close

# results for stem
restem = pd.read_csv(os.path.join(RESULT_DIRECTORY, 'baseline_results2_small.txt'), sep=',')
for c in list(restem.columns):
	restem.rename(columns={c: c.replace(' ', '')}, inplace=True)

restem['White'] = restem['mse_test_stem_white']
restem['Hispanic'] = restem['mse_test_stem_hispanic']
restem['Black'] = restem['mse_test_stem_black']
restem['Asian'] = restem['mse_test_stem_asian']
restem['Female'] = restem['mse_test_stem_f']
restem['Male'] = restem['mse_test_stem_m']

restem.loc['pmf', 'Method'] = 'PMF'
restem.loc['pmf_bias', 'Method'] = 'PMFB'
restem.loc['bpmf', 'Method'] = 'BPMF'
restem.loc['fastfm', 'Method'] = 'FM'
restem.loc['fastfm_features', 'Method'] = 'FMX'

restem = restem.set_index('Method')

restem['Hispanic'].test = abs((restem.mse_test_stem_white - restem.mse_test_stem_hispanic) / (restem.var_test_stem_white + restem.var_test_stem_hispanic) ** 0.5)
restem['Black'].test = abs((restem.mse_test_stem_white - restem.mse_test_stem_black) / (restem.var_test_stem_white + restem.var_test_stem_black) ** 0.5)
restem['Asian'].test = abs((restem.mse_test_stem_white - restem.mse_test_stem_asian) / (restem.var_test_stem_white + restem.var_test_stem_asian) ** 0.5)
restem['Male'].test = abs((restem.mse_test_stem_f - restem.mse_test_stem_m) / (restem.var_test_stem_f + restem.var_test_stem_m) ** 0.5)

stem_export = restem[['White', 'Hispanic', 'Black', 'Asian', 'Female', 'Male']]
stem_export['Hispanic'].test = restem['Hispanic'].test
stem_export['Black'].test = restem['Black'].test 
stem_export['Asian'].test = restem['Asian'].test 
stem_export['Male'].test = restem['Male'].test


with open(os.path.join(RESULT_DIRECTORY, 'mse_stem_results.tex'), 'w') as output:
	output.write("\\" + "begin{table} \n")
	output.write("\\centering  \n")
	output.write("\\label{tab:3}  \n")
	output.write("\\begin{tabular}{c|cccc|cc} \n")
	
	output.write('Recommender' + ''.join(' & %s' %c for c in list(stem_export.columns)) +  '\\' + '\\' + '\n')
	output.write("\\hline \n")
	output.write("\\hline \n")
	
	for meth in baseline_export.index:
		output.write(meth + ''.join(add_significance(stem_export[v], meth) for v in list(stem_export.columns)) +  '\\' + '\\' + '\n')
	
	output.write("\\end{tabular} \n")
	output.write("\\caption{Mean squared errors on the test sample across demographic groups for students in STEM majors. "
				"Values in bold characters show a statistically significant difference at the $95\%$ using a left tail t-distribution. t-test compares mean squared errors relative to White or female students. "
				"Recommender systems are trained using only students with demographic information }" + '\n')
	
	
	output.write("\\end{table} \n")
	
output.close

# experiment: using all data
eresults1 = pd.read_csv(os.path.join(RESULT_DIRECTORY, 'full_results_fmdim.txt'), sep=',')
for c in list(eresults1.columns):
	eresults1.rename(columns={c: c.replace(' ', '')}, inplace=True)

eresults1['All'] = eresults1['mse_test']
eresults1['White'] = eresults1['mse_test_white']
eresults1['Hispanic'] = eresults1['mse_test_hispanic']
eresults1['Black'] = eresults1['mse_test_black']
eresults1['Asian'] = eresults1['mse_test_asian']
eresults1['Female'] = eresults1['mse_test_f']
eresults1['Male'] = eresults1['mse_test_m']

exp_export = eresults1[['All', 'White', 'Hispanic', 'Black', 'Asian', 'Female', 'Male']]

with open(os.path.join(RESULT_DIRECTORY, 'mse_fm_full_results.tex'), 'w') as output:
	output.write("\\" + "begin{table} \n")
	output.write("\\centering  \n")
	output.write("\\label{tab:4}  \n")
	output.write("\\begin{tabular}{c|ccccc|cc} \n")
	
	output.write('Latent space dimension' + ''.join(' & %s' %c for c in list(exp_export.columns)) +  '\\' + '\\' + '\n')
	output.write("\\hline \n")
	output.write("\\hline \n")
	
	for dim in exp_export.index:
		output.write("%d" %dim + ''.join(" & %.3f" %v for v in exp_export.loc[dim, :].values) +  '\\' + '\\' + '\n')
	
	output.write("\\end{tabular} \n")
	output.write("\\caption{Mean squared errors on the test sample across demographic groups using factorization machine with features and the full dataset.}" + '\n')
	
	output.write("\\end{table} \n")
	
output.close


# experiment -- control noise heteroskedasticity - all sample
def add_significance(series, method):

	v = series.loc[method]
	
	if hasattr(series, "test"):
		tv = series.test.loc[method]
		if (tv > 1.645):
			return " & \\textbf{%.3f}" %v
		else:
				return " & %.3f" %v
	else:
		return " & %.3f" %v

enoise = pd.read_csv(os.path.join(RESULT_DIRECTORY, 'demo_results2_all.txt'), sep=',')
for c in list(enoise.columns):
	enoise.rename(columns={c: c.replace(' ', '')}, inplace=True)
	
enoise.loc['no_features', 'features'] = 'NONE' 
enoise.loc['sex_', 'features'] = 'SEX'
enoise.loc['srace_', 'features'] = 'RACE'
enoise.loc['sex_srace_', 'features'] = 'SEX $+$ RACE'
enoise.loc['is_male_', 'features'] = 'IS MALE'
enoise.loc['is_male_is_black_', 'features'] = 'IS BLACK $+$ IS MALE'

enoise['All'] = enoise['mse_test']
enoise['White'] = enoise['mse_test_white']
enoise['Black'] = enoise['mse_test_black']
enoise['Female'] = enoise['mse_test_f']
enoise['Male'] = enoise['mse_test_m']
enoise.loc['is_black_', 'features'] = 'IS BLACK'

enoise = enoise.set_index('features')
enoise['Black'].test = abs((enoise['mse_test_white'] - enoise['mse_test_black']) / (enoise['var_test_black'] + enoise['var_test_white']) ** 0.5)
enoise['Male'].test = abs((enoise['mse_test_f'] - enoise['mse_test_m']) / (enoise['var_test_m'] + enoise['var_test_f']) ** 0.5)

enoise_exp = enoise[['All', 'White', 'Black', 'Female', 'Male']]
enoise_exp['Black'].test = enoise['Black'].test
enoise_exp['Male'].test = enoise['Male'].test


with open(os.path.join(RESULT_DIRECTORY, 'mse_demo_results_all.tex'), 'w') as output:
	output.write("\\" + "begin{table} \n")
	output.write("\\centering  \n")
	output.write("\\label{tab:4}  \n")
	output.write("\\begin{tabular}{c|ccc|cc} \n")
	
	output.write('Demographic features' + ''.join(' & %s' %c for c in list(enoise_exp.columns)) +  '\\' + '\\' + '\n')
	output.write("\\hline \n")
	output.write("\\hline \n")
	
	for method in enoise_exp.index:
		output.write("%s" %method + ''.join(add_significance(enoise_exp[v], method) for v in list(enoise_exp.columns)) +  '\\' + '\\' + '\n')
	
	output.write("\\end{tabular} \n")
	output.write("\\caption{Mean squared errors on the test sample across demographic groups using FMX and adding demographc variables. " 
						"IS MALE is a dummy variable equal one if the student reports to be a male. "
						"IS BLACK is a dummy variable equal to one if the student reports to be black. "
						"The experiment uses all students  in the dataset, including the ones without demographic "
						"information. Values in bold characters show a statistically significant difference with white or female at the $95\%$ level using a left tail t-distribution.}" + '\n')
	
	output.write("\\end{table} \n")
	
output.close

# experiment -- control noise heteroskedasticity -- small sample
enoise1 = pd.read_csv(os.path.join(RESULT_DIRECTORY, 'demo_results2_small.txt'), sep=',')
for c in list(enoise1.columns):
	enoise1.rename(columns={c: c.replace(' ', '')}, inplace=True)
	
	
enoise1.loc['no_features', 'features'] = 'NONE' 
enoise1.loc['sex_', 'features'] = 'SEX'
enoise1.loc['srace_', 'features'] = 'RACE'
enoise1.loc['sex_srace_', 'features'] = 'SEX $+$ RACE'
enoise1.loc['is_male_', 'features'] = 'IS MALE'
enoise1.loc['is_black_', 'features'] = 'IS BLACK'
enoise1.loc['is_male_is_black_', 'features'] = 'IS BLACK $+$ IS MALE'

enoise1['All'] = enoise1['mse_test']
enoise1['White'] = enoise1['mse_test_white']
enoise1['Black'] = enoise1['mse_test_black']
enoise1['Female'] = enoise1['mse_test_f']
enoise1['Male'] = enoise1['mse_test_m']

 
enoise1 = enoise1.set_index('features')
enoise1['Black'].test = abs((enoise1['mse_test_white'] - enoise1['mse_test_black']) / (enoise1['var_test_black'] + enoise1['var_test_white']) ** 0.5)
enoise1['Male'].test = abs((enoise1['mse_test_f'] - enoise1['mse_test_m']) / (enoise1['var_test_m'] + enoise1['var_test_f']) ** 0.5)

enoise_exp1 = enoise1[['All', 'White', 'Black', 'Female', 'Male']]
enoise_exp1['Black'].test = enoise1['Black'].test
enoise_exp1['Male'].test = enoise1['Male'].test


with open(os.path.join(RESULT_DIRECTORY, 'mse_demo_results_small.tex'), 'w') as output:
	output.write("\\" + "begin{table} \n")
	output.write("\\centering  \n")
	output.write("\\label{tab:4}  \n")
	output.write("\\begin{tabular}{c|ccc|cc} \n")
	
	output.write('Demographic features' + ''.join(' & %s' %c for c in list(enoise_exp1.columns)) +  '\\' + '\\' + '\n')
	output.write("\\hline \n")
	output.write("\\hline \n")
	
	for method in enoise_exp1.index:
		output.write("%s" %method + ''.join(add_significance(enoise_exp1[v], method) for v in list(enoise_exp1.columns)) +  '\\' + '\\' + '\n')
	
	output.write("\\end{tabular} \n")
	output.write("\\caption{Mean squared errors on the test sample across demographic groups using FMX and adding demographc variables. " 
						"IS MALE is a dummy variable equal one if the student reports to be a male. "
						"IS BLACK is a dummy variable equal to one if the student reports to be black. "
						"The experiment uses only students with demographic information. Values in bold characters show a statistically significant difference with white or female at the $95\%$ level using a left tail t-distribution.}" + '\n')
	
	output.write("\\end{table} \n")
	
output.close


# predicted by grade
predg = pd.read_csv(".\\results\\baseline_mse_grade_all.csv")
predg = predg[predg.Grade != 1.33]
predg.sort_values(by="Grade", inplace=True)

# rename columns
for c in list(predg.columns):
	predg.rename(columns={c: c.replace(' ', '')}, inplace=True)
predg.columns = map(str.capitalize, predg.columns)
predg.rename(columns={"F":"Female", "M": "Male"}, inplace=True)
predg.set_index('Grade', inplace=True)
predg.fillna(0, inplace=True)

print(predg.columns)
print(predg)
with open(os.path.join(RESULT_DIRECTORY, 'base_mse_results_by_grade.tex'), 'w') as output:
	output.write("\\" + "begin{table} \n")
	output.write("\\centering  \n")
	
	output.write("\\begin{tabular}{c|ccccc|cc} \n")
	
	output.write('Grade' + ''.join(' & %s' %c for c in list(predg.columns) if c!='') +  '\\' + '\\' + '\n')
	output.write("\\hline \n")
	output.write("\\hline \n")
	
	for grade in predg.index:
		output.write("%.2f" %grade + ''.join(" & %s "%predg.loc[grade, v][:5] for v in list(predg.columns) if v!='') +  '\\' + '\\' + '\n')
	
	output.write("\\end{tabular} \n")
	output.write("\\caption{Mean squared errors on the test sample across demographic groups using FMX and across observed grades. " 
						"The experiment uses all students with demographic information, including the ones without demographic information. Values in bold characters show a statistically significant difference with white or female at the $95\%$ level using a left tail t-distribution.}" + '\n')
	output.write("\\label{tab:4}  \n")
	output.write("\\end{table} \n")
	
output.close




