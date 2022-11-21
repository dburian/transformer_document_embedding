export TEXINPUTS=../tex//:

all: thesis.pdf abstract.pdf

# LaTeX must be run multiple times to get references right
thesis.pdf: thesis.tex $(wildcard *.tex) $(wildcard chapters/*.tex) bibliography.bib thesis.xmpdata
	pdflatex $<
	bibtex thesis
	pdflatex $<
	pdflatex $<

abstract.pdf: abstract.tex abstract.xmpdata
	pdflatex $<

clean:
	rm -f *.log *.dvi *.aux *.toc *.lof *.lot *.out *.bbl *.blg *.xmpi
	rm -f chapters/*.aux
	rm -f thesis.pdf abstract.pdf
