<template>
  <div class="base">
      <Input
        :value="getValue(valueDef[name], value)"
        :label="name"
        :type="type"
        :parse="parse"
        :name="name"
        :key="blockType + key"
        inline
        @change="change"
      />
      <Select
        :value="getValue(valueDef[name], value)"
        :label="label"
        :lists="list"
        :parse="parse"
        :name="name"
        :key="blockType + key"
        @change="change"
      />
      <Select
        :value="getValue(valueDef[name], value)"
        :label="label"
        :lists="list"
        :parse="parse"
        :name="name"
        :key="blockType + key"
        @change="change"
      />
  </div>
</template>

<script>
import Input from "@/components/forms/Input.vue";
import Checkbox from "@/components/forms/Checkbox.vue";
import Select from "@/components/forms/Select.vue";

export default {
  name: "Forms",
  components: {
    Input,
    Select,
    Checkbox,
  },
  props: {
    data: {
      type: Object,
      default: () => ({ type: "main", items: [], value: {} }),
    },
  },
  computed: {
    items() {
      return this.data?.items || [];
    },
    valueDef() {
      // console.log(this.data?.value);
      return this.data?.value || {};
    },
    type() {
      return this.data?.type || "";
    },
    blockType() {
      return this.data?.blockType || "";
    },
  },
  methods: {
    change(e) {
      this.$emit("change", { type: this.type, ...e });
    },
    getValue(val, defVal) {
      const value = val ?? defVal;
      if (typeof value === "object") {
        return value.join();
      }
      return value
    },
  },
  filters: {
    toString: function (value) {
      // console.log( value )
      if (typeof value === "object") {
        return value.join();
      }
      return value;
    },
    isCheck: function (value) {
      // console.log( value )
      // console.log( typeof(value) )
      return !!value;
    },
  },
  mounted() {
    this.$emit('height', this.$el.clientHeight)
  }
};
</script>