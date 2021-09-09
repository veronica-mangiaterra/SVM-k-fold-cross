
#csv sources
files = [r'path1',r'path2']

#fetch data in csv and structure in a class:vector dict
cl_dict = get_data(files)
t1 = list(cl_dict.keys())[0]
t2 = list(cl_dict.keys())[1]

data = list(cl_dict[t1].values())+list(cl_dict[t2].values())
my_vectors = np. array(data)
my_labels = make_labels(len(list(cl_dict[t1].values())), len(list(cl_dict[t2].values())))

#setup SVM setup and print output
print('SVC output:')
clf    = SVC(C = C, verbose = True, kernel = kernel, degree = degree) #prints data
print(cross_val_score (SVC(), my_vectors, my_labels, cv=9))
y_pred = cross_val_predict(clf, my_vectors, my_labels)
make_confmat(my_labels,y_pred, 'ita', 'dutch')
