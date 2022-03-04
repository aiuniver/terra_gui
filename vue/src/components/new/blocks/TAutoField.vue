<template>
  <div class="forms">
    <t-multi-select
      v-if="type === 'multiselect_sources_paths'"
      :id="id"
      name="sources_paths"
      label="Выберите путь"
      @multiselect="$emit('multiselect', $event)"
      :errors="error"
      inline
    />
    <t-segmentation-manual
      v-if="type === 'segmentation_manual'"
      :id="id"
      :label="label"
      type="text"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :error="error"
      inline
      @change="change"
      @cleanError="cleanError"
    />
    <t-segmentation-annotation
      v-if="type === 'segmentation_annotation'"
      :value="getValue"
      :label="label"
      :id="id"
      type="text"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :error="error"
      inline
      @change="change"
      @cleanError="cleanError"
    />
    <t-segmentation-search
      v-if="type === 'segmentation_search'"
      :value="getValue"
      :id="id"
      :label="label"
      type="text"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :error="error"
      inline
      @change="change"
      @cleanError="cleanError"
    />
    <TInput
      v-if="type === 'tuple'"
      :value="getValue"
      :label="label"
      type="text"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :error="error"
      inline
      @change="change"
      @cleanError="cleanError"
    />
    <TInput
      v-if="type === 'number' || type === 'text'"
      :value="getValue"
      :label="label"
      :type="type"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :error="error"
      :update="update"
      inline
      @change="change"
      @cleanError="cleanError"
    />
    <Checkbox
      v-if="type === 'checkbox'"
      inline
      :value="getValue"
      :label="label"
      type="checkbox"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :event="event"
      :error="error"
      @cleanError="cleanError"
      @change="change"
    />
    <TSelect
      v-if="type === 'select'"
      :value="getValue"
      :label="label"
      :lists="list"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :error="error"
      @cleanError="cleanError"
      @change="change"
    />
    <t-select-tasks
      v-if="type === 'select_creation_tasks'"
      :value="getValue"
      :label="label"
      :lists="list"
      :parse="parse"
      :name="name"
      :key="name + idKey"
      :error="error"
      @cleanError="cleanError"
      @change="change"
    />
    <template v-for="(data, i) of dataFields">
      <t-auto-field
        v-bind="data"
        :idKey="idKey + i"
        :key="idKey + i"
        :id="id"
        :parameters="parameters"
        :update="update"
        @multiselect="$emit('multiselect', $event)"
        @change="$emit('change', $event)"
      />
    </template>
  </div>
</template>

<script>
import TInput from '@/components/new/forms/TInput';
import Checkbox from '@/components/new/forms/Checkbox';
import TSelect from '@/components/new/forms/Select';

export default {
  name: 't-auto-field',
  components: {
    TInput,
    Checkbox,
    TSelect
  },
  props: {
    idKey: String,
    type: String,
    value: [String, Boolean, Number, Array],
    list: Array,
    event: String,
    label: String,
    parse: String,
    name: String,
    fields: Object,
    id: Number,
    root: Boolean,
    parameters: Object,
    update: Object,
    isAudio: Number,
  },
  data: () => ({
    valueIn: null,
    audio: [
      {
        type: 'select',
        name: 'parameter',
        label: 'Параметр',
        parse: 'parameter',
        value: 'audio_signal',
        disabled: false,
        list: [
          {
            value: 'audio_signal',
            label: 'Audio signal',
          },
          {
            value: 'chroma_stft',
            label: 'Chroma STFT',
          },
          {
            value: 'mfcc',
            label: 'MFCC',
          },
          {
            value: 'rms',
            label: 'RMS',
          },
          {
            value: 'spectral_centroid',
            label: 'Spectral centroid',
          },
          {
            value: 'spectral_bandwidth',
            label: 'Spectral bandwidth',
          },
          {
            value: 'spectral_rolloff',
            label: 'Spectral roll-off',
          },
          {
            value: 'zero_crossing_rate',
            label: 'Zero-crossing rate',
          },
        ],
        fields: null,
        api: null,
      },
    ],
  }),
  computed: {
    getValue() {
      return this.parameters?.[this.name] ?? this.value;
    },
    errors() {
      return this.$store.getters['datasets/getErrors'](this.id);
    },
    error() {
      const key = this.name;
      return this.errors?.[key]?.[0] || this.errors?.parameters?.[key]?.[0] || '';
    },
    dataFields() {
      if (this.name === 'type' && this.valueIn === 'Audio' && this.isAudio !== this.id) {
        return this.audio;
      } else {
        if (!!this.fields && !!this.fields[this.valueIn]) {
          return this.fields[this.valueIn];
        } else {
          return [];
        }
      }
    },
  },
  methods: {
    change({ value, name }) {
      this.valueIn = null;
      this.$emit('change', { id: this.id, value, name, root: this.root });
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
    this.$emit('change', { id: this.id, value: this.getValue, name: this.name, root: this.root });
    // console.log(this.name, this.parameters, this.getValue)
    this.$nextTick(() => {
      this.valueIn = this.getValue;
    });
    this.$emit('height', this.$el.clientHeight);
  },
};
</script>