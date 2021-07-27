<template>
  <div class="params">
    <scrollbar :style="height">
      <div class="params-container">
        <DatasetButton />
        <div class="params-item load-dataset-field">
          <form
            class="inner form-inline-label"
            novalidate="novalidate"
            autocomplete="off"
          >
            <ul class="tabs">
              <li
                :class="{active: tabGoogle} "
                @click.prevent="tabGoogle = true"
              >
                Google drive
              </li>
              <li
                :class="{active: !tabGoogle} "
                @click.prevent="tabGoogle = false"
              >
                URL-ссылка
              </li>
            </ul>
            <Autocomplete
              v-show="tabGoogle"
              :options="items"
              label="Выберите файл из Google-диска"
              @focus="focus"
              @selected="selected"
            />
            <div v-show="!tabGoogle" class="field-form">
              <label>Введите URL на архив исходников</label>
              <input
                v-model="urlName"
                type="text"
              />
            </div>
            <div class="field-form field-inline field-reverse inputs">
              <label for="num_links[inputs]"
                >Кол-во <b>входов</b></label
              >
              <input
                v-model="inputs"
                type="number"
                min="0"
                max="100"
                name="num_links[inputs]"
                id="num_links[inputs]"
                data-value-type="number"
              />
            </div>
            <div class="field-form field-inline field-reverse outputs">
              <label for="num_links[outputs]"
                >Кол-во <b>выходов</b></label
              >
              <input
                type="number"
                name="num_links[outputs]"
                id="num_links[outputs]"
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
                    <!-- <template v-for="(input, i) of inputLayer">
                      <Layer
                        def="images"
                        :parse="`inputs[input_${input}]`"
                        :name="`input_${input}`"
                        :key="'input_' + i"
                      />
                    </template> -->
                  </div>
                </at-collapse-item>
                <at-collapse-item class="mt-3" title="Выходные слои">
                  <div class="inner row inputs-layers">
                    <!-- <template v-for="(output, i) of outputLayer">
                      <Layer
                        def="classification"
                        :parse="`outputs[output_${output}]`"
                        :name="`output_${output}`"
                        :key="'output_' + i"
                      />
                    </template> -->
                  </div>
                </at-collapse-item>
              </at-collapse>
              <div class="params-item dataset-params mt-2">
                <div class="params-title">Параметры датасета</div>
                <div class="inner form-inline-label px-5 py-3">
                  <div class="field-form">
                    <label for="parameters[name]">Название датасета</label>
                    <input id="parameters[name]" type="text" name="parameters[name]" />
                  </div>
                  <div class="field-form">
                    <label for="parameters[user_tags]">Теги</label>
                    <input id="parameters[user_tags]" type="text" name="parameters[user_tags]" />
                  </div>
                  <DatasetSlider />
                  <div class="field-form field-inline field-reverse">
                    <label for="parameters[preserve_sequence]">Сохранить последовательность</label>
                    <div class="checkout-switch">
                      <input
                        type="checkbox"
                        name="parameters[preserve_sequence]"
                        id="parameters[preserve_sequence]"
                      />
                      <span class="switcher"></span>
                    </div>
                  </div>
                  <div class="field-form field-inline field-reverse">
                    <label for="parameters[use_generator]" >Использовать генератор</label>
                    <div class="checkout-switch">
                      <input
                        type="checkbox"
                        name="parameters[use_generator]"
                        id="parameters[use_generator]"
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
    </scrollbar>
  </div>
</template>

<script>

import { mapGetters } from "vuex";
import Autocomplete from "@/components/forms/Autocomplete.vue";
// import Input from "@/components/forms/Input.vue";
import DatasetSlider from "@/components/datasets/params/DatasetSlider.vue";
// import Layer from "./Layer.vue";
import DatasetButton from "./DatasetButton.vue";
import serialize from "@/assets/js/serialize";
export default {
  name: "Settings",
  components: {
    Autocomplete,
    // Layer,
    DatasetButton,
    DatasetSlider,
    // Input
  },
  data: () => ({
    tabGoogle: true,
    urlName: '',
    googleName: '',
    interval: null,
    inputs: 1,
    outputs: 1,
    items: [],
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
    createInterval() {
      this.interval = setInterval(async () => {
        
        const data = await this.$store.dispatch("datasets/loadProgress", {});
        const { finished, message, percent } = data
        if ( !data || finished ) {
          clearTimeout(this.interval);
          this.$store.dispatch("messages/setProgressMessage", message );
          this.$store.dispatch("messages/setProgress", percent);
        } else {
          this.$store.dispatch("messages/setProgress", percent);
          this.$store.dispatch("messages/setProgressMessage",  message );
        }
        console.log(data)
      }, 1000); 
    },
    async download() {
      if (this.googleName) {
        const res = {
          mode: this.tabGoogle ? 'GoogleDrive' : 'URL',
          value: this.tabGoogle ? this.googleName : this.urlName,
        };
        this.createInterval()
        await this.$store.dispatch("datasets/sourceLoad", res);
      } else {
        this.$store.dispatch("messages/setMessage", {
          error: "Выберите файл",
        });
      }
    },
    async focus() {
      const data = await this.$store.dispatch("axios", {
        url: "/datasets/sources/?term=",
      });
      if (!data) {
        return;
      }
      console.log(data)
      this.items = data.map(({ label, value }, i) => {
        return { name: label, id: ++i, value };
      });
    },
    async selected(value) {
      console.log(value)
      this.googleName = value.value;
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

<style lang="scss" scoped>
.params {
  flex-shrink: 0;
  width: 400px;
  border-left: #0e1621 solid 1px;
}
button {
  font-size: 0.875rem;
}
</style>