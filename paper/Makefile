# Makefile for paper compilation
# Usage: make paper   — compile main manuscript
#        make si      — compile supplementary information
#        make all     — compile both
#        make clean   — remove build artefacts

MAIN  := main_manuscript
SI    := supplementary/supplementary_information

.PHONY: all paper si clean

all: paper si

paper: $(MAIN).pdf

si: $(SI).pdf

$(MAIN).pdf: $(MAIN).tex references.bib $(wildcard *.tex revisions/*.tex figures/*.png)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN) || true
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

$(SI).pdf: $(SI).tex $(wildcard supplementary/*.tex)
	cd supplementary && pdflatex -interaction=nonstopmode supplementary_information.tex
	cd supplementary && pdflatex -interaction=nonstopmode supplementary_information.tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).pdf
	rm -f $(SI).aux $(SI).log $(SI).out $(SI).toc $(SI).pdf
	rm -f supplementary/*.aux supplementary/*.log supplementary/*.out supplementary/*.toc
