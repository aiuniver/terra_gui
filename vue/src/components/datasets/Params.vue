<template>
    <div class="properties">
    <div class="wrapper custom-scrollbar-wrapper mCustomScrollbar _mCS_2 mCS_no_scrollbar"><div id="mCSB_2" class="mCustomScrollBox mCS-light mCSB_vertical mCSB_outside" style="max-height: none;" tabindex="0"><div id="mCSB_2_container" class="mCSB_container mCS_y_hidden mCS_no_scrollbar_y" style="position:relative; top:0; left:0;" dir="ltr">
        <div class="params">
            <div class="params-container">
                <div class="params-item dataset-change">
                    <form class="inner" novalidate="novalidate">
                        <div class="actions-form">
                            <div class="item prepare"><button disabled="disabled">Подготовить</button></div>
                            <div class="item delete"><button disabled="disabled">Удалить</button></div>
                            <div class="item change"><button disabled="disabled">Редактировать</button></div>
                        </div>
                    </form>
                </div>
                <div class="params-item load-dataset-field">
                    <form class="inner form-inline-label" novalidate="novalidate">
                        <input type="radio" name="mode" value="google_drive" data-value-type="string" checked="checked">
                        <input type="radio" name="mode" value="url" data-value-type="string">
                        <ul class="tabs">
                            <li data-type="google_drive" class="active">Google drive</li>
                            <li data-type="url">URL-ссылка</li>
                        </ul>
                        <div class="field-form field-mode-type field-mode-google_drive relative">
                            <label for="gdrive-select">Выберите файл из Google-диска</label>
                            <input type="text" name="name" id="gdrive-select" class="jquery-ui-autocomplete ui-autocomplete-input" data-source="/api/v1/datasets-sources/" autocomplete="off">
                        <ul id="ui-id-1" tabindex="0" class="ui-menu ui-widget ui-widget-content ui-autocomplete ui-front" style="display: none;"></ul></div>
                        <div class="field-form field-mode-type field-mode-url hidden">
                            <label for="field_form-link">Введите URL на архив исходников</label>
                            <input type="text" name="link" id="field_form-link" data-value-type="string">
                        </div>
                        <div class="field-form field-inline field-reverse inputs">
                            <label for="field_form-num_links[inputs]">Кол-во <b>входов</b></label>
                            <input type="number" name="num_links[inputs]" id="field_form-num_links[inputs]" data-value-type="number">
                        </div>
                        <div class="field-form field-inline field-reverse outputs">
                            <label for="field_form-num_links[outputs]">Кол-во <b>выходов</b></label>
                            <input type="number" name="num_links[outputs]" id="field_form-num_links[outputs]" data-value-type="number">
                        </div>
                        <div class="actions-form">
                            <div class="item load"><button>Загрузить</button></div>
                        </div>
                    </form>
                </div>
                <div class="params-item dataset-prepare hidden">
                    <form novalidate="novalidate">
                        <div class="params-container">
                            <div class="params-item collapsable">
                                <div class="params-title">Входные слои</div>
                                <div class="inner row inputs-layers">
                                    </div>
                                </div>
                            <div class="params-item collapsable">
                                <div class="params-title">Выходные слои</div>
                                <div class="inner row outputs-layers">
                                </div>
                            </div>
                            <div class="params-item dataset-params">
                                <div class="params-title">Параметры датасета</div>
                                <div class="inner form-inline-label">
                                    <div class="field-form">
                                        <label for="field_form-parameters[name]">Название датасета</label>
                                        <input type="text" name="parameters[name]" id="field_form-parameters[name]" data-value-type="string">
                                    </div>
                                    <div class="field-form">
                                        <label for="field_form-parameters[user_tags]">Теги</label>
                                        <input type="text" name="parameters[user_tags]" id="field_form-parameters[user_tags]" data-value-type="string">
                                    </div>
                                    <div class="field-form field-inline">
                                        <label>Train + val + test</label>
                                        <div class="slider-range ui-slider ui-corner-all ui-slider-horizontal ui-widget ui-widget-content"><div class="ui-slider-range ui-corner-all ui-widget-header" style="left: 59.1398%; width: 32.2581%;"></div><span tabindex="0" class="ui-slider-handle ui-corner-all ui-state-default" style="left: 59.1398%;"></span><span tabindex="0" class="ui-slider-handle ui-corner-all ui-state-default" style="left: 91.3978%;"></span></div>
                                        <input type="number" id="amount1" name="parameters[train_part]" data-value-type="number">
                                        <input type="number" id="amount2" name="parameters[val_part]" data-value-type="number">
                                        <input type="number" id="amount3" name="parameters[test_part]" data-value-type="number">
                                    </div>
                                    <div class="field-form field-inline field-reverse">
                                        <label for="parameters[preserve_sequence]" id="preserve_sequence_label">Сохранить последовательность</label>
                                        <div class="checkout-switch">
                                            <input type="checkbox" id="parameters[preserve_sequence]" name="parameters[preserve_sequence]" data-value-type="boolean" data-unchecked-value="false">
                                            <span class="switcher"></span>
                                        </div>
                                    </div>

                                </div>
                            </div>
                            <div class="params-item">
                                <div class="inner">
                                    <div class="actions-form">
                                        <div class="item prepare"><button>Сформировать</button></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div></div><div id="mCSB_2_scrollbar_vertical" class="mCSB_scrollTools mCSB_2_scrollbar mCS-light mCSB_scrollTools_vertical" style="display: none;"><div class="mCSB_draggerContainer"><div id="mCSB_2_dragger_vertical" class="mCSB_dragger" style="position: absolute; min-height: 30px; height: 0px; top: 0px;"><div class="mCSB_dragger_bar" style="line-height: 30px;"></div></div><div class="mCSB_draggerRail"></div></div></div></div>
</div>
</template>