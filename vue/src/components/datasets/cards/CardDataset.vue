<template>
  <div
    :class="['dataset-card-item', { active: dataset.active, selected: loaded }]"
    @click="$emit('clickCard', dataset, cardIndex)"
  >
    <div class="dataset-card">
      <div class="card-title">{{ dataset.name }}</div>
      <div class="card-body" @click="click">
        <div v-for="({ name }, key) of getFour" :key="`tag_${key}`" class="card-tag">
          {{ name }}
        </div>
      </div>

      <div :class="'card-extra ' + (dataset.size ? 'is-custom' : '')">
        <div class="wrapper">
          <span>
            {{
              dataset.size && dataset.size?.short && dataset.size?.unit
                ? dataset.size.short.toFixed(2) + ' ' + dataset.size.unit
                : 'Предустановленный'
            }}
          </span>
        </div>
        <div class="remove"></div>
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
    getFour() {
      const tags = this.dataset?.tags || [];
      return tags.filter((item, i) => i >= this.index && i < this.index + 4);
    },
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
