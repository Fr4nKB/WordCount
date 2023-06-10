PARAMS = 20 20 1
PARAMS_tree = 5 20 1

run: build
	./wcl $(PARAMS)
	./wcv $(PARAMS)
	./wcm_old $(PARAMS)
	./wcm $(PARAMS)
	./wct $(PARAMS_tree)

build: gen wcl wcv wcm_old wcm wct

wcl: WordCountList.cpp
	g++ WordCountList.cpp -o wcl

wcv: WordCountVector.cpp
	g++ WordCountVector.cpp -o wcv

wcm_old: WordCountMap_old.cpp
	g++ WordCountMap_old.cpp -o wcm_old

wcm: WordCountMap.cpp
	g++ WordCountMap.cpp -o wcm

wct: WordCountTree.cpp
	g++ WordCountTree.cpp -o wct

gen: gen.cpp
	g++ gen.cpp -o gen

clean:
	del wc*.exe gen.exe output*.txt *.png