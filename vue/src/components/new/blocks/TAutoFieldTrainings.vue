<template>
  <div class="forms">
    <TInput
      v-if="type === 'tuple'"
      v-show="visible"
      :value="getValue"
      :label="label"
      type="text"
      :parse="parse"
      :name="name"
      :inline="inline"
      :disabled="disabled"
      @parse="change"
    />
    <t-field v-if="type === 'number' || type === 'text'" v-show="visible" :label="label" inline>
      <TInputNew
        small
        :style="{ width: '90px' }"
        :value="getValue"
        :type="type"
        :parse="parse"
        :name="name"
        :disabled="disabled"
        @parse="change"
      />
    </t-field>
    <t-field v-if="type === 'checkbox'" :label="label" v-show="visible" inline style="margin: 5px 0 0;">
      <TCheckbox :value="getValue" :parse="parse" :name="name" :disabled="disabled" @parse="change" />
    </t-field>
    <t-field v-if="type === 'select'" :label="label" v-show="visible" inline>
      <TSelect
        :value="getValue"
        :list="list"
        :parse="parse"
        :name="name"
        :inline="inline"
        small
        :disabled="disabled"
        @parse="change"
      />
    </t-field>

    <t-field v-if="type === 'auto_complete'" :label="label" v-show="visible">
      <TAutoComplete
        :value="getValue"
        :list="list"
        :parse="parse"
        :name="name"
        :disabled="disabled"
        all
        @parse="change"
      />
    </t-field>
    <MegaMultiSelect
      v-if="type === 'multiselect'"
      v-show="visible"
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
import MegaMultiSelect from '@/components/new/forms/MegaMultiSelect';
import TInput from '@/components/new/forms/TInput';
import TInputNew from '@/components/new/forms/TInputNew';
import TSelect from '@/components/new/forms/TSelect';
import TAutoComplete from '@/components/new/forms/TAutoComplete';
import TCheckbox from '@/components/new/forms/TCheckbox';

export default {
  name: 't-auto-field-trainings',
  components: {
    MegaMultiSelect,
    TInput,
    TSelect,
    TAutoComplete,
    TInputNew,
    TCheckbox
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
    changeable: Boolean,
    visible: Boolean,
    disabled: [Boolean, Array],
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
      this.$emit('parse', { parse, name, value, changeable: this.changeable });
      // this.$nextTick(() => {
      //   this.valueIn = value;
      // });
    },
  },
  created() {
    // console.log(this.disabled);
  },
  mounted() {
    this.$emit('parse', { name: this.name, value: this.getValue, parse: this.parse, changeable: this.changeable, mounted: true });
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
  // padding-top: 10px;
}
</style>