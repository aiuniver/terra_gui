<template>
  <div class="forms">
    <t-tuple-cascade
      v-if="type === 'text_array'"
      :value="getValue"
      :label="label"
      type="text"
      :parse="parse"
      :name="name"
      :error="error"
      inline
      @change="change"
      @cleanError="cleanError"
    />
    <t-input
      v-if="type === 'tuple'"
      :value="getValue"
      :label="label"
      type="text"
      :parse="parse"
      :name="name"
      :error="error"
      inline
      @change="change"
      @cleanError="cleanError"
    />
    <t-input
      v-if="type === 'number' || type === 'text'"
      :value="getValue"
      :label="label"
      :type="type"
      :parse="parse"
      :name="name"
      :error="error"
      :update="update"
      inline
      @change="change"
      @cleanError="cleanError"
    />
    <t-checkbox
      v-if="type === 'checkbox'"
      inline
      :value="getValue"
      :label="label"
      type="checkbox"
      :parse="parse"
      :name="name"
      :event="event"
      :error="error"
      @cleanError="cleanError"
      @change="change"
    />
    <t-field v-if="type === 'select'" :inline="true" :label="label">
      <t-select-new
        :value="getValue"
        :label="label"
        :list="list"
        :parse="parse"
        :name="name"
        inline
        :small="!big"
        :error="error"
        @cleanError="cleanError"
        @change="change"
      />
    </t-field>
    <t-field v-if="type === 'auto_complete'" :label="label">
      <t-auto-complete-new-two :value="getValue" :list="list" :parse="parse" :name="name" all @parse="change" />
    </t-field>
    <template v-for="(data, i) of dataFields">
      <t-auto-field-cascade
        v-bind="data"
        :key="data.name + i"
        :id="id"
        :parameters="parameters"
        :update="update"
        @change="$emit('change', $event)"
      />
    </template>
  </div>
</template>

<script>
export default {
  name: 't-auto-field-cascade',
  props: {
    type: String,
    value: [String, Boolean, Number, Array],
    list: Array,
    event: String,
    label: String,
    parse: String,
    name: String,
    fields: Object,
    manual: Object,
    id: String,
    root: Boolean,
    parameters: Object,
    update: Object,
    isAudio: Number,
    big: Boolean,
  },
  data: () => ({
    valueIn: null,
  }),
  computed: {
    getValue() {
      let val;
      if (this.type === 'select') {
        val = this.list.find(item => item.value === this.parameters?.[this.name])?.value ?? this.value
      } else {
        val = this.parameters?.[this.name] ?? this.value
      }
      return val;
    },
    errors() {
      return this.$store.getters['datasets/getErrors'](this.id);
    },
    error() {
      const key = this.name;
      return this.errors?.[key]?.[0] || this.errors?.parameters?.[key]?.[0] || '';
    },
    dataFields() {
      if (!!this.fields && !!this.fields[this.valueIn]) {
        return this.fields[this.valueIn];
      } else {
        return [];
      }
    },
    info() {
      if (!!this.manual && !!this.manual[this.valueIn]) {
        return this.manual[this.valueIn];
      } else {
        return '';
      }
    },
  },
  methods: {
    change({ value, name }) {
      const block = this.$store.getters['cascades/getBlock']
      setTimeout(() => {
        this.$store.dispatch('cascades/selectBlock', block);
      }, 10);
      console.log(value, name);
      this.valueIn = null;
      this.$emit('change', { id: this.id, value, name, parse: this.parse });
      this.$nextTick(() => {
        this.valueIn = value;
      });
    },
    cleanError() {
      this.$store.dispatch('datasets/cleanError', { id: this.id, name: this.name });
    },
  },
  created() {
    // console.log(this.type)
  },
  mounted() {
    this.$emit('change', { id: this.id, value: this.getValue, name: this.name, mounted: true, parse: this.parse });
    console.log(this.name, this.getValue , this.value);
    // this.valueIn = null;
    this.$nextTick(() => {
      this.valueIn = this.getValue;
    });
  },
};
</script>