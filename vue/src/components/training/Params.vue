<template>
  <div class="properties project-training-properties">
    <div class="wrapper">
      <div class="params">
        <vue-custom-scrollbar
          class="scroll-area"
          :settings="{
            suppressScrollY: false,
            suppressScrollX: true,
            wheelPropagation: false,
          }"
        >
        <form class="params-container">
          <div class="params-item params-config">
            <div class="inner settings">
              <div class="params-item params-optimizer">
                <div class="inner">
                  <div class="field-form field-inline">
                    <label for="optimazer">Оптимизатор</label>
                    <Dropdown
                        :options="items"
                        :disabled="false"
                        :label="Оптимизатор"
                        name="optimazer"
                        :maxItem="10"
                        placeholder="Please select an option"
                        @focus="focus"
                        @selected="selected"
                    >
                    </Dropdown>
                  </div>
                  <div class="field-form form-inline-label">
                    <div class="field-form field-inline">
                      <label for="field_form-batch_sizes">Размер батча</label>
                      <input name="batch_sizes" id="field_form-batch_sizes" type="number" value="1" />
                    </div>
                    <div class="field-form field-inline">
                      <label for="field_form-epochs_count">Количество эпох</label>
                      <input name="epochs_count" id="field_form-epochs_count" type="number" value="1" />
                    </div>
                    <div class="field-form field-inline">
                      <label for="field_form-learning_rate">Learning rate</label>
                      <input name="optimizer[params][main][learning_rate]" id="field_form-learning_rate" type="number" value="" />
                    </div>
                  </div>
                </div>
              </div>

              <div class="params-item params-optimizer-extra collapsable collapsed hidden">
                <div class="params-title">Параметры оптимизатора</div>
                <div class="inner form-inline-label"></div>
              </div>

              <div class="params-item params-output collapsable">
                <div class="params-title">Параметры outputs слоев</div>

                <div class="inner">
                  <div class="title">Слой <b>«1»</b></div>
                  <div class="form-inline-label">
                  </div>
                </div>
              </div>

              <div class="params-item params-checkpoint collapsable">
                <div class="params-title">Чекпоинты</div>
                <div class="inner form-inline-label">
                  <div class="field-form field-inline field-reverse">
                    <label for="checkpoint[monitor][output]">Монитор</label>
                     <Dropdown
                        :options="items"
                        :disabled="false"
                        :label="Монитор"
                        name="checkpoint[monitor][output]"
                        :maxItem="10"
                        placeholder="Please select an option"
                        @focus="focus"
                        @selected="selected"
                    >
                    </Dropdown>
                  </div>
                  <div class="field-form field-inline field-reverse">
                    <label for="checkpoint[indicator]">Indicator</label>
                    <Dropdown
                        :options="items"
                        :disabled="false"
                        :label="Монитор"
                        name="checkpoint[monitor][output]"
                        :maxItem="10"
                        placeholder="Please select an option"
                        @focus="focus"
                        @selected="selected"
                    >
                    </Dropdown>
                  </div>
                  <div class="field-form field-inline field-reverse">
                    <label for="checkpoint[monitor][out_type]">Тип</label>
                    <Dropdown
                        :options="items"
                        :disabled="false"
                        :label="Тип"
                        name="checkpoint[monitor][out_type]"
                        :maxItem="10"
                        placeholder="Please select an option"
                        @focus="focus"
                        @selected="selected"
                    >
                    </Dropdown>
                  </div>
                  <div class="field-form field-inline field-reverse">
                    <label for="checkpoint[mode]">Режим</label>
                    <Dropdown
                        :options="items"
                        :disabled="false"
                        :label="Режим"
                        name="checkpoint[mode]"
                        :maxItem="10"
                        placeholder="Please select an option"
                        @focus="focus"
                        @selected="selected"
                    >
                    </Dropdown>
                  </div>
                  <div class="field-form field-inline field-reverse">
                    <label for="checkpoint[save_best]">Сохранить лучшее</label>
                    <div class="checkout-switch">
                      <input type="checkbox" id="checkpoint[save_best]" name="checkpoint[save_best]"/>
                      <span class="switcher"></span>
                    </div>
                  </div>
                  <div class="field-form field-inline field-reverse">
                    <label for="checkpoint[save_weights]">Сохранить веса</label>
                    <div class="checkout-switch">
                      <input type="checkbox" id="checkpoint[save_weights]" name="checkpoint[save_weights]"/>
                      <span class="switcher"></span>
                    </div>
                  </div>
                </div>
              </div>

              <div class="params-item params-callbacks collapsable">
                <div class="params-title">Выводить</div>
                <div class="inner callback-1">
                  <div class="title">Слой <b>«1»</b></div>
                  <div class="form-inline-label"></div>
                </div>
              </div>
            </div>
          </div>

          <div class="params-item params-actions">
            <div class="inner actions">
              <div class="actions-form">
                <div class="item training">
                  <button>Обучить</button>
                </div>
                <div class="item stop">
                  <button disabled="disabled">Остановить</button>
                </div>
                <div class="item reset">
                  <button disabled="disabled">Сбросить</button>
                </div>
              </div>
            </div>
          </div>
        </form>
        </vue-custom-scrollbar>
      </div>
    </div>
  </div>
</template>

<script>
import vueCustomScrollbar from "vue-custom-scrollbar";
import Dropdown from "@/components/forms/Dropdown.vue";
// import Layer from "@/components/datasets/Layer.vue";

export default {
  name: "Params",
  components: {
    Dropdown,
    // Layer,
    vueCustomScrollbar,
  },
}
</script>

<style scoped>

</style>