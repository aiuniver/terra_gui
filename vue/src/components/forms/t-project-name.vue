<template>
    <div class="t__project">
        <div class="t__project__label">Project:</div>
        <input type="text"
        class="t__project__name"
        v-model="nameProject"
        @blur="saveProject"
        @input="handleInput"
        ref="input"
        >
        <i></i>
    </div>
</template>

<script>
export default {
    name: 't-project-name',
    data() {
        return {
            toSave: false
        }
    },
    computed: {
        nameProject: {
            set(name) {
                this.$store.dispatch('projects/setProject', { name });
            },
            get() {
                return this.$store.getters['projects/getProject'].name;
            },
        }
    },
    methods: {
        handleInput(e) {
            this.toSave = true
            e.target.style.width = `${e.target.value.length+2}ch`
        },
        async saveProject() {
            if (!this.toSave) return
            if (this.nameProject.length > 2) {
                this.$store.dispatch('messages/setMessage', {
                message: `Изменение названия проекта на «${this.nameProject}»`,
                });
                await this.$store.dispatch('projects/saveProject', {
                name: this.nameProject,
                });
                this.$store.dispatch('messages/setMessage', {
                message: `Название проекта изменено на «${this.nameProject}»`,
                });
                this.toSave = false
            } else {
                this.$store.dispatch('messages/setMessage', {
                error: 'Длина не может быть < 3 сим.',
                });
            }
        }
    },
    updated() {
        const input = this.$refs.input
        input.style.width = this.$store.getters['projects/getProject'].name.length + 2 + 'ch'
    }
}
</script>

<style lang="scss" scoped>
.t__project {
display: -webkit-box;
display: -moz-box;
display: -ms-flexbox;
display: -webkit-flex;
display: flex;
-webkit-box-direction: normal;
-moz-box-direction: normal;
-webkit-box-orient: horizontal;
-moz-box-orient: horizontal;
-webkit-flex-direction: row;
-ms-flex-direction: row;
flex-direction: row;
-webkit-flex-wrap: nowrap;
-ms-flex-wrap: nowrap;
flex-wrap: nowrap;
-webkit-box-pack: start;
-moz-box-pack: start;
-webkit-justify-content: flex-start;
-ms-flex-pack: start;
justify-content: flex-start;
-webkit-align-content: flex-start;
-ms-flex-line-pack: start;
align-content: flex-start;
-webkit-box-align: center;
-moz-box-align: center;
-webkit-align-items: center;
-ms-flex-align: center;
align-items: center;
margin-left: 10px;
&__label {
    color: #a7bed3;
    margin: 0 5px 0 0;
    user-select: none;
}
&__name {
    position: relative;
    white-space: nowrap;
    font-weight: 700;
    display: flex;
    align-items: center;
    height: 100%;
    max-width: 300px;
    min-width: 70px;
    border: none;
    padding: 0 5px;
    box-sizing: content-box;
    background: none;
    &:focus {
        border: 1px solid rgb(108, 120, 131);
    }
    + i {
    display: block;
    width: 13px;
    height: 13px;
    margin-left: 5px;
    background-position: center;
    background-repeat: no-repeat;
    background-size: contain;
    background-image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTMiIGhlaWdodD0iMTMiIHZpZXdCb3g9IjAgMCAxMyAxMyIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTAuODc1IDkuNzgyODFWMTIuMTI2NkgzLjMxNjQxTDEwLjUxNjkgNS4yMTQwNkw4LjA3NTUyIDIuODcwMzFMMC44NzUgOS43ODI4MVpNMTIuNDA0OSAzLjQwMTU2QzEyLjY1ODkgMy4xNTc4MSAxMi42NTg5IDIuNzY0MDYgMTIuNDA0OSAyLjUyMDMxTDEwLjg4MTUgMS4wNTc4MUMxMC42Mjc2IDAuODE0MDYyIDEwLjIxNzQgMC44MTQwNjIgOS45NjM1NCAxLjA1NzgxTDguNzcyMTMgMi4yMDE1NkwxMS4yMTM1IDQuNTQ1MzFMMTIuNDA0OSAzLjQwMTU2WiIgZmlsbD0iI0E3QkVEMyIvPgo8L3N2Zz4K);
    }
}
}
</style>