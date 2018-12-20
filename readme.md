# 凸优化课程大作业：L1-TWSVM 读书报告

## git 命令

+ `clone`：将远程库下载至本地，推荐 ssh 方式，这样 push 不需要输入密码。
+ `checkout`：切换分支。`-b` 选项建立新分支。每人的内容均新建 tex 文件，并且在自己新建的分支进行作业。
+ `branch`：分支相关，查看、新建、删除等。
+ `pull`：拉取远程更新。可将队友提交的更新拉取至本地，按上述，其内容在另一个分支。
+ `push`：推送本地更新到远程库。
+ `stash`：暂存未提交的修改，得以切换分支查看队友进度。
+ `reset`：回滚至历史节点。
+ `add`：添加文件至工作区。
+ `commit`：提交修改。
    + 提交时尽量不要包括不必要的文件
+ `status`：查看工作区状态。
+ `log`：查看提交历史。
+ `reflog`：查看分支切换以及回滚操作历史。


**Latex模板中新增的包**
```
\usepackage{float}
\usepackage{subfigure}
\usepackage{caption}
```
**对模板中设置的一点点改动**
原来`\setcounter{page}{1} \setcounter{section}{#4} \noindent`
修改后`\setcounter{page}{1} \setcounter{section}{0} \noindent`
