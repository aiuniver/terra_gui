<template>
  <div class="properties">
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
          <div class="params-container">
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
                <Dropdown
                  :options="items"
                  :disabled="false"
                  :label="
                    tabGoogle
                      ? 'Выберите файл из Google-диска'
                      : 'Введите URL на архив исходников'
                  "
                  name="zipcode"
                  :maxItem="10"
                  placeholder="Please select an option"
                  @focus="focus"
                  @selected="selected"
                >
                </Dropdown>
                <div class="field-form field-mode-type field-mode-url hidden">
                  <label for="field_form-link"
                    >Введите URL на архив исходников</label
                  >
                  <input
                    type="text"
                    name="link"
                    id="field_form-link"
                    data-value-type="string"
                  />
                </div>
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
                  <at-collapse>
                    <at-collapse-item class="mt-3" title="Входные слои">
                      <div class="inner row inputs-layers">
                        <template v-for="(input, i) of imputLayer">
                          <Layer :name="`input_${input}`" :key="i" />
                        </template>
                      </div>
                    </at-collapse-item>
                    <at-collapse-item class="mt-3" title="Выходные слои">
                      <div class="inner row inputs-layers">
                        <!-- <template v-for="(input, i) of imputLayer"> -->
                          <!-- <Layer :key="i" /> -->
                        <!-- </template> -->
                      </div>
                    </at-collapse-item>
                  </at-collapse>

                  <!-- <div class="params-item collapsable">
                    <div class="params-title">Входные слои</div>
                    <div class="inner row inputs-layers">
                      <template v-for="(input, i) of +inputs">
                        <Layer :settings="settings" :key="i" />
                      </template>
                    </div>
                  </div> -->
                  <!-- <div class="params-item collapsable">
                    <div class="params-title">Выходные слои</div>
                    <div class="inner row outputs-layers">
                      <Layer :settings="settings" :key="i" />
                    </div>
                  </div> -->
                </div>
              </form>
            </div>
            <div class="mt-5"><button @click.prevent="click">test</button></div>
          </div>
        </vue-custom-scrollbar>
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import vueCustomScrollbar from "vue-custom-scrollbar";
import Dropdown from "@/components/forms/Dropdown.vue";
import Layer from "@/components/datasets/Layer.vue";
import serialize from "@/assets/js/serialize";
export default {
  name: "Settings",
  components: {
    Dropdown,
    Layer,
    vueCustomScrollbar,
  },
  data: () => ({
    tabGoogle: true,
    options: [
      { id: 1, name: "Option 1" },
      { id: 2, name: "Option 2" },
    ],
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
      settings: 'datasets/getSettings'
    }),
    imputLayer() {
      const int = +this.inputs
      const settings = this.settings
      console.log(settings)
      return ((int > 0 && int < 100) && Object.keys(settings).length) ? int : 0;
    },
  },
  methods: {
    async download() {
      if (this.name && this.inputs && this.outputs) {
        this.isLoading = true;
        try {
          const res = {
            name: this.name,
            inputs: this.inputs,
            outputs: this.outputs,
          };
          await this.$store.dispatch("datasets/settings", res);
          this.$Notify.success({ title: "success", message: 'success' });
        } catch (error) {
          this.$Notify.error({ title: "Error", message: error });
        }
        this.isLoading = false;
      } else {
        this.$Notify.error({ title: "Error", message: "Check file" });
      }
    },
    async focus() {
      if (!this.items.length) {
        const res = {
          method: "get",
          url: "/api/v1/datasets-sources/?term=",
        };
        const data = await this.$store.dispatch("datasets/axios", res);
        this.items = data.map((item, i) => {
          return { name: item.value, id: ++i };
        });
      }
    },
    async selected(value) {
      this.name = value.name;
    },
    click() {
      console.log(this.$refs.form.$el);

      if (this.$refs.form) {
        console.log(
          serialize(this.$refs.form, {
            hash: true,
            disabled: true,
            empty: true,
          })
        );
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
  height: 600px;
}
</style>