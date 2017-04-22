VIARGS=Makefile foo nn-punk.py ansi.py {RO}../page-deform-keras/page-dedeform.py ../findimg/findimg.py
VISTART=nn-punk.py
SHELL=/bin/bash

all: do


do:
	stdbuf -i0 -o0 ./nn-punk.py 2>&1 | tee foo

view:
	stdbuf -i0 -o0 ./nn-punk.py -v -e 2>&1 | tee foo

unbend:
	stdbuf -i0 -o0 ./nn-punk.py -c 2>&1 | tee foo

hyperenhance:
	stdbuf -i0 -o0 ./nn-punk.py 2>&1 | tee foo

enhance:
	stdbuf -i0 -o0 ./nn-punk.py -e 2>&1 | tee foo

profile:
	stdbuf -i0 -o0 python3.5 -m cProfile -s cumtime ./nn-punk.py 2>&1 | tee foo

vi:
	@#s=`for f in $(VIARGS); do if [[ "$$f" =~ ^\{RO\} ]]; then printf "%s " "-c 'arge +set\ ro $${f#\{RO\}}'"; else printf "%s " "-c 'arge $$f'"; fi; done; printf "%s" "-c 'arge $(VISTART)'"`; printf "%s " $$s; eval "vim $$s"
	a=(); for f in $(VIARGS); do if [[ "$$f" =~ ^\{RO\} ]]; then a+=('-c' "arge +set\ ro $${f#\{RO\}}"); else a+=('-c' "arge $$f"); fi; done; a+=('-c' 'arge $(VISTART)'); printf "%s\n" "$${a[@]}"; vim "$${a[@]}"
	@#vim $(VIARGS)

clean:
	rm data/weights*
	#rm weights-*

ctags:
	ctags nn-punk.py
