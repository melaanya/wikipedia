#include <string>
#include <functional>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <set>
#include <limits>
#include <math.h>
#include <list>
#include <time.h>
#include <stdlib.h>
#include <valarray>
#include <algorithm>
#include <omp.h>
#include <sys/time.h>
#include "macros.cpp"
//#include <boost/math/distributions/hypergeometric.hpp>


using namespace std;

const int NUM_OF_DOCS = 2365436;  //  //742tk 
const int NUM_OF_CATEGS = 500000;
const int NUM_OF_TERMS = 2085166;
const int NUM_OF_PAIRS = 10000000;  // 1000000
const int NUM_OF_TEST_DOCS = 452168;  // на один больше, с учетом того, что нулевой не использукм
const int NUM_OF_EXPERIMENTS = 1;
const double EPS = 0.00001;
class Category
{
public:
	int numofthecategory;
	int numofdocs;
	vector <int> docs;
	map <int, int> terms;
};

class Document
{
public:
	int numofthedoc;
	map <int, int> terms;
	valarray <int> fastTerms, fastCounts;
	set <int> categories;
	double getSimilarity(Document & d);  // min	
	double getSimilarityKnn(Document & d);  // min
	int categoryfrom;
};

struct EntropiaPair
{
	int doc;
	int categ;
	double entropia;
};

struct InfoMetrics
{
	double metrics;
	int cat, doc;
	bool belong;

	bool operator==(InfoMetrics const & rhs) const
	{
		return abs(metrics - rhs.metrics) < EPS;
	}
};

bool compare_bigger(InfoMetrics lhs, InfoMetrics rhs)
{
	return lhs.metrics > rhs.metrics;
}

bool compare_less(InfoMetrics lhs, InfoMetrics rhs)
{
	return lhs.metrics < rhs.metrics;
}

vector <Document> docs(NUM_OF_DOCS + 1);
vector <Document> testdocs(NUM_OF_TEST_DOCS + 1);
vector <Category> categs(NUM_OF_CATEGS);
vector <InfoMetrics> metrics(NUM_OF_PAIRS);
vector <int> randDocs(NUM_OF_EXPERIMENTS);
map <int, int> frequency;
map <int, list <int>> featTestDocs;

void percent(double& lastProcent, double newProcent, clock_t t0, bool& flagAlreadyPrintTime)
{
	//dP - через сколько процентов выводим новое значение
	double dP = 0.1;
	clock_t t1 = clock();
	double toc = (double)(t1 - t0) / CLOCKS_PER_SEC;
	if (((newProcent>0.01) || (toc>60)) && (!flagAlreadyPrintTime))
	{
		flagAlreadyPrintTime = true;
		double rab2 = toc / newProcent;
		if (rab2<60) cout << "EstT = " << rab2 << " sec. ";
		else
			if (rab2<3600) cout << "EstT = " << rab2 / 60 << " min. ";
			else cout << "EstT = " << rab2 / 3600 << " hours. ";
			cout.flush();
	}
	if (newProcent - lastProcent >= dP)
	{
		lastProcent = newProcent;
		cout << floor(newProcent * 100) << "% ";
		cout.flush();
		if (1 - lastProcent < dP) cout << endl;
	}
}
double Document::getSimilarityKnn(Document & d)
{
	int i1 = 0, i2 = 0;
	int n1 = fastTerms.size();
	int n2 = d.fastTerms.size();
	int common = 0;
	if (n1 <= 0 || n2 <= 0)
	{
		//cout << "n1 <= 0 || n2 <= 0" << n1 << " " << n2 << " " << numofthedoc << " " << d.numofthedoc << endl;
		return 0;
		//exit(1);
	}
	while (true)
	{
		if (fastTerms[i1] < d.fastTerms[i2])
		{
			while (i1 != n1 && fastTerms[i1] < d.fastTerms[i2])
			{
				i1++;
			}
			if (i1 == n1) break;
		}
		else
		{
			while (i2 != n2 && fastTerms[i1] > d.fastTerms[i2])
			{
				i2++;
			}
			if (i2 == n2) break;
		}
		if (fastTerms[i1] == d.fastTerms[i2])
		{
			common++;
			++i1; ++i2;
		}
		if (i1 == n1 || i2 == n2) break;
	}
	return (n1 + n2 - 2.0 * common) / (n1 + n2 - common);
}


double Document::getSimilarity(Document & d)
{
	double res = 0;
	int i1 = 0, i2 = 0;
	int n1 = fastTerms.size();
	int n2 = d.fastTerms.size();
	if (n1 <= 0 || n2 <= 0)
	{
		//cout << "n1 <= 0 || n2 <= 0" << n1 << " " << n2 << " " << numofthedoc << " " << d.numofthedoc << endl;
		return 0;
		//exit(1);
	}
	while (true)
	{
		if (fastTerms[i1] < d.fastTerms[i2])
		{
			while (i1 != n1 && fastTerms[i1] < d.fastTerms[i2])
			{
				i1++;
			}
			if (i1 == n1) break;
		}
		else
		{
			while (i2 != n2 && fastTerms[i1] > d.fastTerms[i2])
			{
				i2++;
			}
			if (i2 == n2) break;
		}
		if (fastTerms[i1] == d.fastTerms[i2])
		{
			//double toadd = min(fastCounts[i1], d.fastCounts[i2]);
			res += 1;
			++i1; ++i2;
		}
		if (i1 == n1 || i2 == n2) break;
	}
	return res / this->terms.size();
}

// чтение и обработка исходного файла
void read_all_docs_without_catterms(ifstream & fin)
{
	int k = 0;
	fin.ignore(numeric_limits<streamsize>::max(), '\n');
	double lastProcent = 0;
	clock_t t0 = clock();
	bool flagAlreadyPrintTime = false;
	while (!fin.eof())
	{
		Document newdoc;
		newdoc.numofthedoc = k;
		char ch = ',';
		while (ch != ' ' && !fin.eof())
		{
			int a;
			fin >> a;
			newdoc.categories.insert(a);
			categs[a].numofthecategory = a;
			++categs[a].numofdocs;
			fin.get(ch);
			categs[a].docs.push_back(k); // номер документа
		}
		while (ch == ' ')
		{
			int feat, value;
			fin >> feat;
			fin.get(ch);
			fin >> value;
			newdoc.terms[feat] = value;
			fin.get(ch);
		}
		newdoc.fastTerms.resize(newdoc.terms.size());
		newdoc.fastCounts.resize(newdoc.terms.size());
		int i = 0;
		for (auto it = newdoc.terms.begin(); it != newdoc.terms.end(); ++it)
		{
			newdoc.fastTerms[i] = it->first;
			newdoc.fastCounts[i] = it->second;
			++i;
		}
		docs[k] = newdoc;
		k++;
		percent(lastProcent, (double)k / NUM_OF_DOCS, t0, flagAlreadyPrintTime);
	}
	docs.pop_back();  //удал€ю последний, т.к. он некорректно считываетс€
}

void read_test_file(ifstream &fin)
{
	int k = 1;
	double lastProcent = 0;
	clock_t t0 = clock();
	bool flagAlreadyPrintTime = false;
	while (!fin.eof())
	{
		Document newdoc;
		newdoc.numofthedoc = k;
		char ch = ','; // произвольна€ инициализаци€
		while (ch != ' ' && !fin.eof())
			fin.get(ch);
		while (ch == ' ' && !fin.eof())
		{
			int feat, value;
			fin >> feat;
			ROBOT_ASSERT_EXT1(feat < NUM_OF_TERMS, feat);
			//terms[feat]++;
			fin.get(ch);
			fin >> value;
			newdoc.terms[feat] = value;
			featTestDocs[feat].push_back(k);
			fin.get(ch);
		}
		newdoc.fastTerms.resize(newdoc.terms.size());
		newdoc.fastCounts.resize(newdoc.terms.size());
		int i = 0;
		for (auto it = newdoc.terms.begin(); it != newdoc.terms.end(); ++it)
		{
			newdoc.fastTerms[i] = it->first;
			newdoc.fastCounts[i] = it->second;
			++i;
		}
		testdocs[k] = newdoc;
		k++;			
		percent(lastProcent, (double)k / NUM_OF_TEST_DOCS, t0, flagAlreadyPrintTime);
	}
}


bool reverseCatComp_big(pair <double, int> lhs, pair <double, int> rhs)
{
	return (lhs.first > rhs.first) || ((lhs.first == rhs.first) && (lhs.second > rhs.second));
}


double entropia(double q0, double q1)
{
	if (q0 == 0 || q1 == 0)
		return 0;
	return -q0*log2(q0) - q1*log2(q1);
}

double entropia2(double p, double n)
{
	return entropia(p / (p + n), n / (p + n));
}
int alpha0, alpha1;

bool continuePred(int ii, double entropy, double entropyBarrier)
{
	if (ii < alpha0) return true;
	if (ii > alpha1) return false;
	return entropy >= entropyBarrier;
}

void entropiaForDoc(FILE * fout, int doc, int chosendocs, vector <Document> & curDocs, int kindOfSimilarityFunction, bool trainFile)
{
	map <int, int> commonCategs;
	vector <InfoMetrics> docMetrics(NUM_OF_DOCS);
	for (int j = 0; j < NUM_OF_DOCS; ++j)
	{
		if (docs[j].numofthedoc < 0 || trainFile && j == doc)   // пропускаю несуществующие документы
		{
			docMetrics[j].doc = -1;
			continue;
		}
		if (kindOfSimilarityFunction == 1)
			docMetrics[j].metrics = curDocs[doc].getSimilarity(docs[j]);
		if (kindOfSimilarityFunction == 2)
			docMetrics[j].metrics = curDocs[doc].getSimilarityKnn(docs[j]);   
		docMetrics[j].doc = j;
	}
	if (kindOfSimilarityFunction == 1)
		sort(docMetrics.begin(), docMetrics.end(), compare_bigger);
	if (kindOfSimilarityFunction == 2)
		sort(docMetrics.begin(), docMetrics.end(), compare_less);
	double totalDocs = 0;
	for (int j = 0; j < chosendocs; ++j)  // chosendocs - число выбранных соседних документов
	{
		int curDoc = docMetrics[j].doc;
		ROBOT_ASSERT(curDoc != -1);
		for (auto it = docs[curDoc].categories.begin(); it != docs[curDoc].categories.end(); ++it)
		{
			if (commonCategs.find(*it) == commonCategs.end())  // если еще не добавл€ли ничего
				totalDocs += categs[*it].numofdocs; 
			commonCategs[*it]++; // ищем общие категории
		}
	}
	for (auto it = commonCategs.begin(); it != commonCategs.end(); ++it)
	{
		// по-хорошему надо проверить формулу вычислени€ энтропии еще раз
		EntropiaPair temp;
		temp.doc = doc;
		temp.categ = it->first;
		double l = it->second;
		double k = categs[it->first].numofdocs;
		double p = l, n = chosendocs - l, P = k, N = totalDocs - k;
		double ifEntropia = (p + n) / (P + N) * entropia2(p, n) + (P + N - p - n) / (P + N) * entropia2(P - p, N - n);
		double iGain = entropia2(P, N) - ifEntropia;
		if (l / k < k / totalDocs)  // если условна€ веро€тность меньше безусловной
			iGain = 0;
		temp.entropia = iGain;
#pragma omp critical
		fwrite(&temp, sizeof(EntropiaPair), 1, fout);  
	}
}


void fileTotalEntropia(FILE * fout, int chosendocs, vector <Document> & curDocs)
{
	int done = 0;
	double t00 = clock();
#pragma omp parallel for reduction(+:done)
	for (int i = 1; i < curDocs.size(); ++i)
	{
		if (curDocs[i].numofthedoc == -1000000000)
			continue;
		entropiaForDoc(fout, i, chosendocs, curDocs, 1, false);
		++done;
#pragma omp critical
		if (i % 10000 == 0)
		{
			double curTime = (double)(omp_get_wtime() - t00) / 3600;
			cout << "done = " << done << ", time = " << curTime << ", remain = " << curTime * (curDocs.size() - done) / done << endl;
		}
	}
}


int main()
{
	srand(1);
	ifstream fin, fintest;
	fin.open("train.csv");
	fintest.open("test.csv");

	for (int i = 0; i < NUM_OF_DOCS; ++i)
	{
		docs[i].numofthedoc = -1000000000;
	}

	for (int i = 0; i < NUM_OF_CATEGS; ++i)
	{
		categs[i].numofthecategory = -1000000000;
	}

	//	read_all_docs(fin);
	read_all_docs_without_catterms(fin);
	cout << "finish processing train file" << endl;
	
	read_test_file(fintest);

	// вывод информации об энтропии пары документ-категори€ в файл
	int chosendocs = 30;  // было дл€ 30
	/*FILE * fout;
	if ((fout = fopen("/media/local_hdd/ankaberg/entropiaForTrain.bin", "wb"))== NULL) 
	{
		printf("ошибка при открытии файла \n");
		exit(1);
	}*/
	//fileTotalEntropia(fout, chosendocs);
	
	/*FILE * foutKnn;
	if ((foutKnn = fopen("/media/local_hdd/ankaberg/entropiaForTrainKnnMetrics.bin", "wb"))== NULL) 
	{
		printf("ошибка при открытии файла \n");
		exit(1);
	}*/
	
	FILE * foutTestFile;
	if ((foutTestFile = fopen("/media/local_hdd/ankaberg/entropiaForTestCheckout.bin", "wb"))== NULL) 
	{
		printf("ошибка при открытии файла \n");
		exit(1);
	}
	
	//fileTotalEntropia (foutKnn, chosendocs);
	fileTotalEntropia(foutTestFile, chosendocs, testdocs);
	//fileTotalEntropiaTest(foutTestFile, chosendocs, foutNumOfPairs);
	
	cout << "finsihed writing" << endl;
	//fclose(foutKnn);
	//fclose(foutTestFile);
	
	//fclose(fout);


	return 0;
}