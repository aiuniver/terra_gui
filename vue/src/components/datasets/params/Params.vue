<template>
  <div class="params">
    <scrollbar :style="height">
      <div class="params__items">
        <div class="params__items--item">
          <DatasetButton />
        </div>
        <div class="params__items--item pa-0">
          <DatasetTab @select="select" />
        </div>
        <div class="params__items--item py-0">
          <DatasetTwoInput />
        </div>
        <div class="params__items--item">
          <button @click.prevent="download">Загрузить</button>
        </div>
        <form novalidate="novalidate" ref="form">
          <at-collapse value="0">
            <at-collapse-item title="Входные слои">
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
            <at-collapse-item title="Выходные слои">
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
          <div class="params__items--title">Параметры датасета</div>
          <div class="params__items--item">
            <div class="inner form-inline-label">
              <div class="field-form">
                <label for="parameters[name]">Название датасета</label>
                <input
                  id="parameters[name]"
                  type="text"
                  name="parameters[name]"
                />
              </div>
              <div class="field-form">
                <label for="parameters[user_tags]">Теги</label>
                <input
                  id="parameters[user_tags]"
                  type="text"
                  name="parameters[user_tags]"
                />
              </div>
              <DatasetSlider />
              <div class="field-form field-inline field-reverse">
                <label for="parameters[preserve_sequence]"
                  >Сохранить последовательность</label
                >
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
                <label for="parameters[use_generator]"
                  >Использовать генератор</label
                >
                <div class="checkout-switch">
                  <input
                    type="checkbox"
                    name="parameters[use_generator]"
                    id="parameters[use_generator]"
                  />
                  <span class="switcher"></span>
                </div>
              </div>
              <button class="mt-6" @click.prevent="click">Сформировать</button>
            </div>
          </div>
        </form>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
import DatasetTwoInput from "@/components/datasets/params/DatasetTwoInput.vue";
import DatasetSlider from "@/components/datasets/params/DatasetSlider.vue";
import DatasetTab from "@/components/datasets/params/DatasetTab.vue";
// import Layer from "./Layer.vue";
import DatasetButton from "./DatasetButton.vue";
import serialize from "@/assets/js/serialize";
export default {
  name: "Settings",
  components: {
    DatasetTab,
    // Layer,
    DatasetButton,
    DatasetSlider,
    DatasetTwoInput,
  },
  data: () => ({
    dataset: {},
    interval: null,
    inputs: 1,
    outputs: 1,
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
        const { finished, message, percent } = data;
        if (!data || finished) {
          clearTimeout(this.interval);
          this.$store.dispatch("messages/setProgressMessage", message);
          this.$store.dispatch("messages/setProgress", percent);
        } else {
          this.$store.dispatch("messages/setProgress", percent);
          this.$store.dispatch("messages/setProgressMessage", message);
        }
        console.log(data);
      }, 1000);
    },
    select(select) {
      console.log(select)
      this.dataset = select
    },
    async download() {
      const { mode, value } = this.dataset
      if (mode && value) {
        this.createInterval()
        await this.$store.dispatch("datasets/sourceLoad", { mode, value });
      } else {
        this.$store.dispatch("messages/setMessage", {
          error: "Выберите файл",
        });
      }
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
  width: 400px;
  flex-shrink: 0;
  border-left: #0e1621 solid 1px;
  // border-left: #0e1621  1px solid;
  &__items {
    &--item {
      padding: 20px;
    }
    &--title {
      display: block;
      line-height: 1.25;
      margin: 0 0 10px 0;
      padding: 5px 20px;
      font-size: 0.75rem;
      user-select: none;
      background-color: #0e1621;
    }
  }
}
button {
  font-size: 0.875rem;
}
</style>