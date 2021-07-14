<template>
  <div class="properties project-training-properties">
    <div class="wrapper">
      <div class="params">
        <form class="params-container">
          <div class="params-item params-config">
            <div class="inner settings">
              <div class="params-item params-optimizer pa-3">
                <div class="inner">
                  <div class="field-form field-inline">
                    <Autocomplete
                      :options="optimazer_items"
                      :disabled="false"
                      :label="'Оптимизатор'"
                      name="optimazer"
                      :maxItem="10"
                      placeholder="Please select an option"
                      @focus="focus"
                      @selected="selected"
                    >
                    </Autocomplete>
                  </div>
                  <div class="field-form form-inline-label flex wrap">
                    <div class="field-form field-inline">
                      <label for="field_form-batch_sizes">Размер батча</label>
                      <input
                        name="batch_sizes"
                        id="field_form-batch_sizes"
                        type="number"
                        value="1"
                      />
                    </div>
                    <div class="field-form field-inline">
                      <label for="field_form-epochs_count"
                        >Количество эпох</label
                      >
                      <input
                        name="epochs_count"
                        id="field_form-epochs_count"
                        type="number"
                        value="1"
                      />
                    </div>
                    <div class="field-form field-inline">
                      <label for="field_form-learning_rate"
                        >Learning rate</label
                      >
                      <input
                        name="optimizer[params][main][learning_rate]"
                        id="field_form-learning_rate"
                        type="number"
                        value=""
                      />
                    </div>
                  </div>
                </div>
              </div>

              <at-collapse>
                <at-collapse-item class="mt-3" title="Параметры оптимизатора">
                  <div class="form-inline-label">
                    <div class="field-form field-inline field-reverse">
                      <label for="field_form-optimizer[params][extra][beta_1]"
                        >Beta 1</label
                      >
                      <input
                        type="number"
                        id="field_form-optimizer[params][extra][beta_1]"
                        name="optimizer[params][extra][beta_1]"
                        value="0.9"
                        data-value-type="number"
                      />
                    </div>
                    <div class="field-form field-inline field-reverse">
                      <label for="field_form-optimizer[params][extra][beta_2]"
                        >Beta 2</label
                      >
                      <input
                        type="number"
                        id="field_form-optimizer[params][extra][beta_2]"
                        name="optimizer[params][extra][beta_2]"
                        value="0.999"
                        data-value-type="number"
                      />
                    </div>
                    <div class="field-form field-inline field-reverse">
                      <label for="field_form-optimizer[params][extra][epsilon]"
                        >Epsilon</label
                      >
                      <input
                        type="number"
                        id="field_form-optimizer[params][extra][epsilon]"
                        name="optimizer[params][extra][epsilon]"
                        value="1e-7"
                        data-value-type="number"
                      />
                    </div>
                    <div class="field-form field-inline field-reverse">
                      <label for="field_form-optimizer[params][extra][amsgrad]"
                        >Amsgrad</label
                      >
                      <div class="checkout-switch">
                        <input
                          type="checkbox"
                          id="field_form-optimizer[params][extra][amsgrad]"
                          name="optimizer[params][extra][amsgrad]"
                          data-value-type="boolean"
                          data-unchecked-value="false"
                        />
                        <span class="switcher"></span>
                      </div>
                    </div>
                  </div>
                </at-collapse-item>

                <at-collapse-item class="mt-3" title="Параметры выходных слоев">
                </at-collapse-item>

                <at-collapse-item class="mt-3" title="Чекпоинты">
                  <div class="inner form-inline-label flex wrap">
                    <Select
                      :label="'Монитор'"
                      :lists="select_list"
                      :value="'OPTION_1'"
                      :parse="'checkpoint[monitor][output]'"
                      :name="'checkpoint[monitor][output]'"
                    />
                    <Select
                      :label="'Indicator'"
                      :lists="select_list"
                      :value="'OPTION_1'"
                      :parse="'checkpoint[indicator]'"
                      :name="'checkpoint[indicator]'"
                    />
                    <Select
                      :label="'Тип'"
                      :lists="select_list"
                      :value="'OPTION_1'"
                      :parse="'checkpoint[monitor][out_type]'"
                      :name="'checkpoint[monitor][out_type]'"
                    />
                    <Select
                      :label="'Режим'"
                      :lists="select_list"
                      :value="'OPTION_1'"
                      :parse="'checkpoint[mode]'"
                      :name="'checkpoint[mode]'"
                    />
                    <Checkbox
                      :value="true"
                      :label="'Сохранить лучшее'"
                      type="checkbox"
                      :parse="'checkpoint[save_best]'"
                      :name="'checkpoint[save_best]'"
                    />
                    <Checkbox
                      :value="true"
                      :label="'Сохранить веса'"
                      type="checkbox"
                      :parse="'checkpoint[save_weights]'"
                      :name="'checkpoint[save_weights]'"
                    />
                  </div>
                </at-collapse-item>

                <at-collapse-item class="mt-3" title="Выводить">
                </at-collapse-item>
              </at-collapse>
            </div>
          </div>

          <div class="params-item params-actions pa-3">
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
      </div>
    </div>
  </div>
</template>

<script>
import Autocomplete from "@/components/forms/Autocomplete.vue";
import Select from "@/components/forms/Select.vue";
import Checkbox from "@/components/forms/Checkbox.vue";

export default {
  name: "Params",
  components: {
    Autocomplete,
    Select,
    Checkbox,
  },
  data: () => ({
    optimazer_items: [
      { id: 1, name: "Adam" },
      { id: 2, name: "SGD" },
    ],
    select_list: ["OPTION_1", "OPTION_2"],
  }),
  methods: {
    focus() {},
    selected() {},
  },
};
</script>

<style scoped>
.form-inline-label {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
}
</style>