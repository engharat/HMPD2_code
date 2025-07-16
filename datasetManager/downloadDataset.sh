#!/bin/bash


d='./' # unzip directory
url=https://cnrsc-my.sharepoint.com/:u:/g/personal/marco_delcoco_cnr_it/EXUUUm7sZgNEn_mfV3WVt4EBLF8zDyrn42s6UbdvvG_R-w?download=1
f='HMPD-full.zip'
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background

wait # finish background tasks