<template>
      <div class="params">
        <vue-custom-scrollbar
          class="scroll-area"
          :settings="scroll"
          :style="height"
        >
          <div class="params-container">
            <DatasetButton @click="click" />
            <div class="params-item load-dataset-field">
              <form
                class="inner form-inline-label"
                novalidate="novalidate"
                autocomplete="off"
              >
                <ul class="tabs">
                  <li
                    :class="tabGoogle ? 'active' : ''"
                    @click.prevent="tabGoogle = true"
                  >
                    Google drive
                  </li>
                  <li
                    :class="!tabGoogle ? 'active' : ''"
                    @click.prevent="tabGoogle = false"
                  >
                    URL-ссылка
                  </li>
                </ul>
                <Autocomplete
                  :options="items"
                  :disabled="false"
                  :label="
                    tabGoogle
                      ? 'Выберите файл из Google-диска'
                      : 'Введите URL на архив исходников'
                  "
                  :maxItem="10"
                  placeholder="Please select file"
                  @focus="focus"
                  @selected="selected"
                />
                <div class="field-form field-inline field-reverse inputs">
                  <label for="field_form-num_links[inputs]"
                    >Кол-во <b>входов</b></label
                  >
                  <input
                    v-model="inputs"
                    type="number"
                    min="0"
                    max="100"
                    name="num_links[inputs]"
                    data-value-type="number"
                  />
                </div>
                <div class="field-form field-inline field-reverse outputs">
                  <label for="field_form-num_links[outputs]"
                    >Кол-во <b>выходов</b></label
                  >
                  <input
                    type="number"
                    name="num_links[outputs]"
                    min="0"
                    max="100"
                    data-value-type="number"
                    :value="outputs"
                  />
                </div>
                <div class="actions-form">
                  <div class="item load">
                    <button @click.prevent="download">Загрузить</button>
                  </div>
                </div>
              </form>
            </div>
            <div class="params-item dataset-prepare">
              <form novalidate="novalidate" ref="form">
                <div class="params-container">
                  <at-collapse value="0">
                    <at-collapse-item class="mt-3" title="Входные слои">
                      <div class="inner row inputs-layers">
                        <template v-for="(input, i) of inputLayer">
                          <Layer
                            def="images"
                            :parse="`inputs[input_${input}]`"
                            :name="`input_${input}`"
                            :key="'input_' + i"
                          />
                        </template>
                      </div>
                    </at-collapse-item>
                    <at-collapse-item class="mt-3" title="Выходные слои">
                      <div class="inner row inputs-layers">
                        <template v-for="(output, i) of outputLayer">
                          <Layer
                            def="classification"
                            :parse="`outputs[output_${output}]`"
                            :name="`output_${output}`"
                            :key="'output_' + i"
                          />
                        </template>
                      </div>
                    </at-collapse-item>
                  </at-collapse>
                  <div class="params-item dataset-params mt-2">
                    <div class="params-title">Параметры датасета</div>
                    <div class="inner form-inline-label px-5 py-3">
                      <div class="field-form">
                        <label>Название датасета</label>
                        <input type="text" name="parameters[name]" />
                      </div>
                      <div class="field-form">
                        <label>Теги</label>
                        <input type="text" name="parameters[user_tags]" />
                      </div>
                      <DatasetSlider />
                      <div class="field-form field-inline field-reverse">
                        <label>Сохранить последовательность</label>
                        <div class="checkout-switch">
                          <input
                            type="checkbox"
                            name="parameters[preserve_sequence]"
                          />
                          <span class="switcher"></span>
                        </div>
                      </div>
                      <button class="mt-6" @click.prevent="click">
                        Сформировать
                      </button>
                    </div>
                  </div>
                </div>
              </form>
            </div>
          </div>
        </vue-custom-scrollbar>
      </div>
</template>

<script>
import { mapGetters } from "vuex";
import vueCustomScrollbar from "vue-custom-scrollbar";
import Autocomplete from "@/components/forms/Autocomplete.vue";
import DatasetSlider from "@/components/datasets/params/DatasetSlider.vue";
import Layer from "./Layer.vue";
import DatasetButton from "./DatasetButton.vue";
import serialize from "@/assets/js/serialize";
export default {
  name: "Settings",
  components: {
    Autocomplete,
    Layer,
    vueCustomScrollbar,
    DatasetButton,
    DatasetSlider,
  },
  data: () => ({
    tabGoogle: true,
    slider: 10,
    options: [
      { id: 1, name: "Option 1" },
      { id: 2, name: "Option 2" },
    ],
    scroll: {
      suppressScrollY: false,
      suppressScrollX: true,
      wheelPropagation: false,
    },
    inputs: 1,
    outputs: 1,
    tab: null,
    name: "",
    items: [],
    isLoading: false,
    search: null,
    rules: {
      length: (len) => (v) => (v || "").length >= len || `Length < ${len}`,
      required: (len) => len.length !== 0 || `Not be empty`,
    },
  }),
  computed: {
    ...mapGetters({
      settings: "datasets/getSettings",
      height: "settings/autoHeight",
    }),
    inputLayer() {
      const int = +this.inputs;
      const settings = this.settings;
      return int > 0 && int < 100 && Object.keys(settings).length ? int : 0;
    },
    outputLayer() {
      const int = +this.outputs;
      const settings = this.settings;
      return int > 0 && int < 100 && Object.keys(settings).length ? int : 0;
    },
  },
  methods: {
    async download() {
      if (this.name && this.inputs && this.outputs) {
        const res = {
          name: this.name,
          inputs: this.inputs,
          outputs: this.outputs,
        };
        await this.$store.dispatch("datasets/settings", res);
      } else {
        this.$store.dispatch("messages/setMessage", { error: "Выберите файл и кол. вход и выходов" });
      }
    },
    async focus() {
      const data = await this.$store.dispatch("axios", {
        url: "/datasets/sources/?term=",
      });
      if (!data) {
        return;
      }
      this.items = data.map(({ label }, i) => {
        return { name: label, id: ++i };
      });
    },
    async selected(value) {
      this.name = value.name;
    },
    click() {
      if (this.$refs.form) {
        const data = serialize(this.$refs.form);
        console.log({ dataset_dict: data });
      } else {
        this.$store.dispatch("messages/setMessage", {
          error: "Error validate",
        });
      }
    },
  },
};
</script>

<style>
.scroll-area {
  position: relative;
  width: 100%;
  /* height: calc(100vh - 152px); */
}
button {
  font-size: 0.875rem;
}
</style>