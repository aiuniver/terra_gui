<template>
  <div class="forms">
    <t-input
      v-if="type === 'tuple'"
      :value="getValue"
      :label="label"
      type="text"
      :parse="parse"
      :name="name"
      :inline="inline"
      :disabled="disabled"
      @parse="change"
    />
    <t-input
      v-if="type === 'number' || type === 'text'"
      :value="getValue"
      :label="label"
      :type="type"
      :parse="parse"
      :name="name"
      :inline="inline"
      :disabled="disabled"
      @parse="change"
    />
    <t-checkbox
      v-if="type === 'checkbox'"
      :value="getValue"
      :label="label"
      type="checkbox"
      :parse="parse"
      :name="name"
      :inline="inline"
      :disabled="disabled"
      @parse="change"
    />
    <t-select-new
      v-if="type === 'select'"
      :value="getValue"
      :label="label"
      :list="list"
      :parse="parse"
      :name="name"
      :inline="inline"
      :disabled="disabled"
      @parse="change"
    />
    <t-auto-complete
      v-if="type === 'auto_complete'"
      :value="getValue"
      :label="label"
      :list="list"
      :parse="parse"
      :name="name"
      :inline="inline"
      :disabled="disabled"
      @parse="change"
    />
    <MegaMultiSelect
      v-if="type === 'multiselect'"
      :value="getValue"
      :label="label"
      :list="list"
      :parse="parse"
      :name="name"
      :inline="inline"
      :disabled="disabled"
      @parse="change"
    />
  </div>
</template>

<script>
import MegaMultiSelect from '@/components/global/forms/MegaMultiSelect';
export default {
  name: 't-auto-field-trainings',
  components: {
    MegaMultiSelect,
  },
  props: {
    type: String,
    value: [String, Boolean, Number, Array],
    list: Array,
    event: String,
    label: String,
    parse: String,
    name: String,
    id: Number,
    state: Object,
    inline: Boolean,
    disabled: Boolean
  },
  data: () => ({
    valueIn: null,
  }),
  computed: {
    getValue() {
      return this.state?.[this.parse] ?? this.value;
    },
    errors() {
      return this.$store.getters['datasets/getErrors'](this.id);
    },
    error() {
      const key = this.name;
      return this.errors?.[key]?.[0] || this.errors?.parameters?.[key]?.[0] || '';
    },
  },
  methods: {
    change({ parse, name, value }) {
      // console.log(parse, value)
      // this.valueIn = null;
      this.$emit('parse', { parse, name, value});
      // this.$nextTick(() => {
      //   this.valueIn = value;
      // });
    },
  },
  created() {
    console.log(this.disabled)
  },
  mounted() {
    this.$emit('parse', { name: this.name, value: this.getValue, parse: this.parse });
    // console.log(this.name, this.parameters, this.getValue)
    this.$nextTick(() => {
      this.valueIn = this.getValue;
    });
    // this.$emit('height', this.$el.clientHeight);
  },
};
</script>

<style lang="scss" scoped>
.forms {
}
</style>