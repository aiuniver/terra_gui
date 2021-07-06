<template>
  <div class="layout-item input-layout">
    <div class="layout-title">
      Слой <b>«{{ name }}»</b>
    </div>
    <div class="layout-params form-inline-label">
      <div class="field-form field-inline field-reverse">
        <label>Название входа</label>
        <input type="text" :name="`inputs[${name}][name]`" :value="name" />
      </div>
      <Select
        v-model="selectType"
        label="Тип данных"
        :lists="settings"
        :value="def"
        :parse="`inputs[${name}][tag]`"
        name="tag"
      />
    </div>
    <Forms :prefixParse="`inputs[${name}]`" :items="items" />
  </div>
</template>

<script>
import Select from "@/components/forms/Select.vue";
import Forms from "@/components/forms";

import { mapGetters } from "vuex";
export default {
  name: "layer",
  components: {
    Select,
    Forms,
  },
  props: {
    name: {
      type: String,
      required: true,
    },
    def: {
      type: String,
      required: true,
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
  mounted() {},
};
</script>