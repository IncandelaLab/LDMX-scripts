# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
alias ll='ls -lh'
alias lt='ls -lht'
alias la='ls -a'
alias lla='ls -la'
alias eamcs='emacs -nw'
alias emacs='emacs -nw'

alias rm~='rm *~'
alias cd..='cd ..'
alias filecount='ls -F | grep -v / | wc -l'

alias cdhome='cd /home/aminali'

alias mybatch='squeue -u aminali'
alias myprior='squeue --format="%.18i %.9p %.8j %.8u %.2t %.19S %.6D %20Y %R" --user=aminali'
alias anav2.3.0Kaons="export LDMX_ANALYSIS_IMG=/home/aminali/ldmx_ana_v2.3.0Kaons.sif"
