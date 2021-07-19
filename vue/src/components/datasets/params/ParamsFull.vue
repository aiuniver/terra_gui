<template>
  <div class="row params">
    <div class="col-24 params__top">
      <div class="row">
        <div
          @drop="onDrop($event, 1)"
          @dragover.prevent
          @dragenter.prevent
          class="col-4 params__top--left"
          draggable="true"
        >
          <h4 draggable="true" >Text</h4>
          <h4 draggable="true" >Text</h4>
          <h4 draggable="true" >Text</h4>
        </div>
        <div
          @dragstart="onDragStart($event, 2)"
          class="col-20 params__top--rigth"
          draggable="true"
        >
          <CardFile name="sdsd" />
          <CardFile  name="bvvb" />
        </div>
      </div>
    </div>
    <div class="col-12 d-flex justify-end pa-3">
      <div class="row">
        <!-- <CardFormInput /> -->
      </div>
    </div>
    <div class="col-12 pa-3">
      <div class="row">
        <!-- <CardFormInput /> -->
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
// import CardFormInput from "@/components/datasets/CardFormInput.vue";
import CardFile from "@/components/datasets/CardFile.vue";
export default {
  name: "Settings",
  components: {
    // CardFormInput,
    CardFile,
  },
  data: () => ({
    items: [
      {
        id: 0,
        title: "Audi",
        categoryId: 0,
      },
      {
        id: 1,
        title: "BMW",
        categoryId: 0,
      },
      {
        id: 2,
        title: "Cat",
        categoryId: 1,
      },
    ],
    categories: [
      {
        id: 0,
        title: "Cars",
      },
      {
        id: 1,
        title: "Animals",
      },
    ],
  }),
  computed: {
    ...mapGetters({
      settings: "datasets/getSettings",
    }),
  },
  methods: {
    onDragStart(e, item) {
            console.log(e)
      console.log(item)
      // e.dataTransfer.dropEffect = "move";
      // e.dataTransfer.effectAllowed = "move";
      // e.dataTransfer.setData("itemId", item.id.toString());
    },
    onDrop(e, categoryId) {

      console.log(e)
      console.log(categoryId)
      // const itemId = parseInt(e.dataTransfer.getData("itemId"));
      // this.items = this.items.map((x) => {
      //   if (x.id == itemId) x.categoryId = categoryId;
      //   return x;
      // });
    },
  },
};
</script>


<style lang="scss">
.full {
  width: 1000px;
}
@import "@/at-ui/scss/variables/index.scss";

.params {
  position: relative;

  &__top {
    padding: 10px;
    background-color: #1b242e;
    &--rigth {
      display: flex;
      justify-content: center;
      padding: 8px;
    }
  }
}
</style>