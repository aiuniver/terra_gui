<template>
  <div class="layout-item input-layout">
    <div class="layout-title">Слой <b>«{{ name }}»</b></div>
    <div class="layout-params form-inline-label">
      <div class="field-form field-inline field-reverse">
        <label for="field_form-inputs[input_1][name]">Название входа</label>
        <input
          type="text"
          :name="`inputs[${name}][name]`"
          :value="name"
        />
      </div>
        <Select
          v-model="selectType"
          label="Тип данных"
          :lists="settings"
          value="images"
          :parse="`inputs[${name}][tag]`"
          name="tag"
        />
    </div>
    <div class="layout-parameters form-inline-label">
      <template v-for="({ type, default: def, available, event }, key) of items">
        <Input
          v-if="type === 'int' || type === 'string'"
          :value="def"
          :label="key"
          :type="type === 'int' ? 'number' : 'text'"
          :parse="`inputs[${name}][parameters][${key}]`"
          :name="key"
          :key="key"
        />
        <Checkbox
          v-if="type === 'bool'"
          :value="def"
          :label="key"
          type="checkbox"
          :parse="`inputs[${name}][parameters][${key}]`"
          :name="key"
          :event="event"
          :key="key"
        />
        <Select
          v-if="available"
          :label="key"
          :lists="available"
          :value="def"
          :parse="`inputs[${name}][parameters][${key}]`"
          :name="key"
          :key="key"
        />
      </template>
    </div>
  </div>
</template>

<script>
import Input from "@/components/forms/Input.vue";
import Checkbox from "@/components/forms/Checkbox.vue";
import Select from "@/components/forms/Select.vue";
import { mapGetters } from "vuex";
export default {
  name: "layer",
  components: {
    Input,
    Select,
    Checkbox
  },
  props: {
    name: {
      type: String,
    },
  },
  data: () => ({
    selectType: "images",
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
      console.log(this.settings)
      return (this.settings || {})[this.selectType] || {};
    },
  },
  methods: {
    change(v) {
      console.log(v);
    },
  },
  mounted() {

  },
};
</script>