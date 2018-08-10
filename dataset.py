import pandas as pd
import os
import numpy as np

DATA_SOURCES = {"courses": "course2015_2017_ALL.csv",
				"students": "students2015_2017_ALL.csv",
				"demographics": "demographics2015_2017_ALL.csv"}
				
COLMAPS = {
    'courses': {
        'sid': ['id'],
        'cid': ['DISC', 'CNUM', 'HRS'],
        'iid': ['INSTR_LNAME', 'INSTR_FNAME'],
        'termnum': ['TERMBNR'],
        'iclass': ['class'],
        'irank': ['instr_rank'],
        'itenure': ['instr_tenure'],
        'cdisc': ['DISC'],
    },
    'demographics': {
        'srace': ['race'],
        'sex': ['SEX']
    },
    'students': {
        'major': ['PMAJR'],
		'cohort': ['cohort']
    }
}

class dataset(object):

	def __init__(self):
		self.register()
	
	def add_table(self, table, tablename):
		setattr(self, tablename, table)

	def register(self):
		# register data from data sources
		for src_name, fname in DATA_SOURCES.items():
			table = pd.read_csv(os.path.join(".\data", fname))
			self.add_table(table, src_name)
					
	def map_col(self, how):
		
		courses_cols = ['id', 'TERMBNR', 'DISC', 'CNUM', 'GRADE', 'HRS',
                        'grdpts', 'INSTR_LNAME', 'INSTR_FNAME', 'class',
						'instr_rank', 'instr_tenure']
		courses = self.courses[courses_cols]
		
		# add students data
		students = self.students
		print(students.columns)
		data = pd.merge(courses, students, 
						on=['id', 'TERMBNR'], 
						how='inner')
						
		# remove NAN grades
		data = data[~np.isnan(data.grdpts)] 
								
		# add demographics data
		demographics = self.demographics
		if(how == 'smaller') | (how is None):
			data = pd.merge(data, demographics, on='id', how='inner')
		elif how == 'all':
			data = pd.merge(data, demographics, on='id', how='left')
			
		# add unknown tags for nan demo data
		data['race'] = data['race'].fillna('UNKNOWN')
		data['SEX'] = data['SEX'].fillna('N')
			
		
		# transform in numerical values categorical data	
		data['iid'] = data[COLMAPS['courses']['iid']].sum(axis=1).astype('category').cat.codes
		data['cid'] = data[COLMAPS['courses']['cid']].astype(str).sum(axis=1).astype('category').cat.codes
		data['sid'] = data['id'].astype('category').cat.codes
		
		features_list = []
		for src_name, coldic in COLMAPS.items():
			for colname, collist in coldic.items():
				if (colname != 'cid') & (colname != 'iid'):
					data[colname] = data[collist[0]].astype('category').cat.codes
					code_min = data[colname].min()
					data[colname] = data[colname] - code_min
				features_list.append(colname)
				
		# add a flag for stem major
		stem_list =['CS', 'INFT', 'FRSC', 'BIOL', 'CHEM', 
               'CPE', 'BIOE', 'PSYC', 'NEUR', 'ASTR', 'ESC',
               'CEIE', 'ACCT', 'ME', 'EVSS', 'PHYS', 'GAME',
               'AOES', 'ACS', 'MLAB', 'CYSE', 'MATH', 'ELEN', 'ISOM']
		data['in_stem'] = (data.COHORT_PMAJR.isin(stem_list)).astype('int32')
				
		# add grade evaluation
		gradevals = ['grdpts']
		
		# add dummy for minority group
		data['is_black'] = (data.race == "BLACK").astype('int32')
		data['is_male'] = (data.SEX == "MALE").astype('int32')
		
		features_list = features_list + gradevals
		return data
		
			
			
if __name__ == "__main__":
	dset = dataset()
	data = dset.map_col(how='smaller')
	print(list(set(data.grdpts)))
	data.to_csv('.\data\preprocessed_students.csv')
	
	