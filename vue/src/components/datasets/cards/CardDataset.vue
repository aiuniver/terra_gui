<template>
  <div :class="['dataset-card-item', {active: dataset.active, selected: loaded}]" @click="$emit('clickCard', dataset, cardIndex)">
    <div class="dataset-card">
      <div class="card-title">{{ dataset.name }}</div>
      <div class="card-body">
        <div v-for="({  name }, key) of dataset.tags" :key="`tag_${key}`" class="card-tag">
          {{ name }}
        </div>
      </div>
      <div :class="'card-extra ' + (dataset.size ? 'is-custom' : '')">
        <div class="wrapper">
          <span>{{ dataset.size ? dataset.size : 'Предустановленный' }}</span>
        </div>
          <div class="remove">            
          </div>
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
    }
  },
};
</script>
