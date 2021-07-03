<template>
  <div class="layout-item input-layout" name="input_1" id="input_1">
    <div class="layout-title">Слой <b>«input_1»</b></div>
    <div class="layout-params form-inline-label">
      <div class="field-form field-inline field-reverse">
        <label for="field_form-inputs[input_1][name]">Название входа</label>
        <input
          type="text"
          id="field_form-inputs[input_1][name]"
          name="inputs[input_1][name]"
          value="input_1"
          data-value-type="string"
        />
      </div>
      <div class="field-form field-inline field-reverse">
        <label for="field_form-inputs[input_1][tag]-button">Тип данных</label>
        <at-select
          v-model="selectType"
          clearable
          size="small"
          style="width: 100px"
          @on-change="change"
        >
          <at-option value="images">images</at-option>
          <at-option value="text">text</at-option>
          <at-option value="audio">audio</at-option>
          <at-option value="dataframe">dataframe</at-option>
        </at-select>
      </div>
    </div>
    <div class="layout-parameters form-inline-label">
      <template v-for="({ type, default:def, available }, key) of items">
        <Input v-if="type==='int' || type==='string'" :value="def" :label="key" :key="key" />
        <Select v-if="available" label="Trest" :lists="available" :value="def" :key="key" />
      </template>
    </div>
  </div>
</template>

<script>
import Input from "@/components/forms/Input.vue";
import Select from "@/components/forms/Select.vue";
import { mapGetters } from "vuex";
export default {
  name: "layer",
  components: {
    Input,
    Select,
  },
  props: {
    name: {
      type: String,
    },
  },
  data: () => ({
    selectType: "",
    rules: {
      length: (len) => (v) => (v || "").length >= len || `Length < ${len}`,
      required: (len) => len.length !== 0 || `Not be empty`,
    },
  }),
  computed: {
    ...mapGetters({
      settings: "datasets/getSettings",
    }),
    items() {
      // console.log(this.selectType)
      // console.log(this.settings)
      return  (this.settings || {})[this.selectType] || {}
    }
  },
  methods: {
    change(v) {
      console.log(v);
    },
  },
  mounted() {
    this.$nextTick(() => {
      const { images } = { ...this.settings };
      if (images) {
        for (const key in images) {
          if (images[key].default) {
            this[key] = images[key].default;
          }
          if (images[key].type) {
            this[key + "_type"] =
              images[key].type === "int" ? "number" : "text";
          }
          if (images[key].available) {
            this[key + "_available"] = images[key].available;
          }
        }
      }
    });
  },
};
</script>