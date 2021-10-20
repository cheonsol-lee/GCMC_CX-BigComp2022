# Knowledge Structure Ver 1.0														
### Copyright(c) 2021.01.28 All rights reserved by Knowledge Innovation Research Center
	Last modified in 2020.11

### DESCRIPTION
This program is composed of 8 packages. 
	- Main.core.sample
	- Main.config
	- Main.nlp
	- Main.coTable
	- Main.model
	- Main.output
	- option.Main.core.output.localgraph
	- option.Main.core.output.webgraph

and each packages contains source codes of conceptual seperation. 

	1) [Main.core.sample] contains the very main method to run this program.
	2) [Main.config] contains a configuration loader, while config.properties contains all the local settings to run the program.
	3) [Main.nlp] analyses the input text and seperate each terms considering Korean and English.
	4) [Main.coTable] calculates cotable of cooccurence and cosign-similarity, sentence-wise and paragraph-wise.
	5) [Main.model] contains source codes which generates Proximity Model with Pathfinder algorithm.
	6) [Main.output] contains textual output generator. The texts are represented by json format 1, json format 2, list of edges, and list of vertexes.
	7) [option.Main.core.output.localgraph] contains jar-based local graph generator exploiting the result.
	8) [option.Main.core.output.webgraph] contains html-based web graph generator exploiting the result.


### INSTALLATION
	no installation is required to run this software since it will never be released as it is. 

### DEPENDENCIES
##### Graph Library
	Web) vis.zip, http://visjs.org/index.html#download_install
	Local) JPathfinder.jar, http://interlinkinc.net/

##### NLP Library
	Korean) KOMORAN, http://www.shineware.co.kr
	English) stanford-postagger-3.6.0.jar, http://nlp.stanford.edu/software/tagger.shtml
	

### Knowledge Graph Generator
[Readme : Knowledge Graph Generator](https://github.com/cheonsol-lee/knowledge_structure_kirc/blob/master/Readme(jupyter).md)

### Knowledge Graph Generator(Neo4j version.)
[Readme : Knowledge Graph Generator(Neo4j)](https://github.com/cheonsol-lee/knowledge_structure_kirc/blob/master/Readme(neo4j).md)

### LICENSE
	Copyright(c)2020 All rights reserved by Knowledge Innovation Research Center

### CREDIT
	Soyoung Cho
	Cheonsol Lee
	Moojin Kim
	JAYONG KIM
 	Jun Woo Kim
	WON EUI HONG
 	JUN YOUNG PARK
 	SANSUNG KIM
 	Sung Chan Kim
 	JUNG HUN KIM
 	KYUNG JIN KIM
 	DONG HWAN BAE
 	HYUNG WOO KIM
