<template>
  <div class="layout-item input-layout">
    <div class="layout-title">
      Слой <b>«{{ name }}»</b>
    </div>
    <div class="layout-params form-inline-label">
      <Input
        label="Название"
        type="text"
        :parse="`${parse}[name]`"
        name="name"
        :value="name"
      />
      <Select
        v-model="select"
        label="Тип данных"
        :lists="settings"
        :parse="`${parse}[tag]`"
        name="tag"
      />
    </div>
    <Forms :parse="parse" :items="items" />
  </div>
</template>

<script>
import Input from "@/components/forms/Input.vue";
import Select from "@/components/forms/Select.vue";
import Forms from "@/components/forms";

import { mapGetters } from "vuex";
export default {
  name: "layer",
  components: {
    Select,
    Forms,
    Input,
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
    parse: {
      type: String,
      required: true,
    },
  },
  data: () => ({
    select: "",
  }),
  computed: {
    ...mapGetters({
      settings: "datasets/getSettings",
    }),
    items() {
      return (this.settings || {})[this.select] || {};
    },
  },
  mounted() {
    console.log(this.def);
    this.select = this.def;
   console.log(this.items);
    // this.$nextTick(() => {
    //   this.select = this.def;
    // });
  },
};
</script>