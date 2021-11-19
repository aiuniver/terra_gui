<template>
  <div :class="['dataset-card-item', { active: dataset.active && !loaded, selected: loaded}]" :title="dataset.name">
    <div :class="['dataset-card', {'disabled': dataset.training_available ? !dataset.training_available : false }]" @click.stop="$emit('click', dataset, cardIndex)">
      <div class="card-title">{{ dataset.name }}</div>
      <div class="card-body" @click="click">
        <div v-if="dataset.tags.length <= 4">
          <div v-for="({ name }, key) of dataset.tags" :key="`tag_${key}`" class="card-tag">
            {{ name }}
          </div>
        </div>
        <scrollbar
          v-else
          :ops="{
            bar: {
              background: '#17212b',
            },
          }"
        >
          <div v-for="({ name }, key) of dataset.tags" :key="`tag_${key}`" class="card-tag">
            {{ name }}
          </div>
        </scrollbar>
      </div>

      <div :class="'card-extra ' + (dataset.size ? 'is-custom' : '')">
        <div class="wrapper">
          <span>
            {{ dataset.size ? dataset.size.short.toFixed(2) + ' ' + dataset.size.unit : 'Предустановленный' }}
          </span>
        </div>

        <div class="remove" @click.stop="$emit('remove', dataset)"></div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    dataset: {
      type: Object,
      default: () => {},
    },
    cardIndex: {
      type: Number,
    },
    loaded: {
      type: Boolean,
      default: false,
    },
  },
  data: () => ({
    index: 0,
  }),
  computed: {
    // getFour() {
    //   const tags = this.dataset?.tags || [];
    //   return tags.filter((item, i) => i >= this.index && i < this.index + 4);
    // },
  },
  methods: {
    click() {
      if (this.dataset.tags.length > this.index + 4) {
        this.index = this.index + 4;
      } else {
        this.index = 0;
      }
    },
  },
};
</script>
