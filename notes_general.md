# epoch

An epoch, in Machine Learning, is the entire processing by the learning algorithm of the entire train-set.

The MNIST train set is composed by 55000 samples. Once the algorithm processed all those 55000 samples an epoch is passed.

=> a full iteration of samples

# venv/ in git
한 번 venv 안에 있는 파일들을 git 에 추가했더니, rm -rf --cached 로 지워도 계속 따라다니길래
git filter-branch --index-filter "git rm -rf --cached --ignore-unmatch 'coursera_datascience/Introduction-Tensorflow(deeplearning.ai)/venv'" -f
이렇게 삭제함.
