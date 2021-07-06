<template>
    <header class="flexbox-center-nowrap px-2">
    <div class="project flexbox-left-nowrap">
        <a href="/project/datasets/" class="logo"></a>
        <div class="title flexbox-left-nowrap">
            <div class="label">Project:</div>
            <div class="name">
                <div class="value flexbox-center-nowrap"><span>NoName</span><i></i></div>
            </div>
        </div>
    </div>
    <div class="name-experiment">
        Название задачи / Название эксперимента
    </div>
    <div class="user flexbox-right-nowrap">
        <div class="item project project-new" data-type="project_new">
            <div class="icon" title="Создать новый проект"><i class="icon-project-new"></i></div>
        </div>
        <div class="item project project-save" data-type="project_save">
            <div class="icon" title="Сохранить проект"><i class="icon-project-save"></i></div>
        </div>
        <div class="item project project-load" data-type="project_load">
            <div class="icon" data-type="project_new" title="Загрузить проект"><i class="icon-project-load"></i></div>
        </div>
        <div class="item profile">
            <div class="icon"><i></i></div>
            <div class="menu">
                <div class="group">
                    <ul>
                        <li><span>Сменить цветовую схему</span></li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
</header>
</template>