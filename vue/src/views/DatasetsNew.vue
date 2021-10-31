<template>
    <main class="page-datasets">
		<div class="cont">
			<div class="datasets-menu">
				<div class="datasets-menu__items">
					<!-- <div class="datasets-menu__items--item"><i class="ci-icon ci-clock"></i> <span>Недавние</span></div>
					<div class="datasets-menu__items--item"><i class="ci-icon ci-file_blank_outline"></i> <span>Проектные</span></div>
					<div class="datasets-menu__items--item"><i class="ci-icon ci-world"></i> <span>Terra</span></div> -->
					<label class="datasets-menu__items--item" :class="{ selected: selectedType === 0 }">
						<input type="radio" name="datasets" v-model="selectedType" :value="0">
						<i class="ci-icon ci-clock"></i><span>Недавние</span>
					</label>
					<label class="datasets-menu__items--item" :class="{ selected: selectedType === 1 }">
						<input type="radio" name="datasets" v-model="selectedType" :value="1">
						<i class="ci-icon ci-file_blank_outline"></i><span>Проектные</span>
					</label>
					<label class="datasets-menu__items--item" :class="{ selected: selectedType === 2 }">
						<input type="radio" name="datasets" v-model="selectedType" :value="2">
						<i class="ci-icon ci-world"></i><span>Terra</span>
					</label>
				</div>
				<hr />
				<div class="datasets-menu__categories">
					<ul class="datasets-menu__categories--item">Изображения
						<li>Машины</li>
						<li>Круглые</li>
						<li>Самолеты</li>
						<li>Квадраты</li>
					</ul>
					<ul class="datasets-menu__categories--item">Видео
						<li>Машины</li>
						<li>Круглые</li>
						<li>Самолеты</li>
						<li>Квадраты</li>
					</ul>
					<ul class="datasets-menu__categories--item">Текст
						<li>Машины</li>
						<li>Круглые</li>
						<li>Самолеты</li>
						<li>Квадраты</li>
					</ul>
				</div>
			</div>
			<Dataset :datasets="datasets" :selectedType="selectedType"/>
		</div>
    </main>
</template>

<script>
import { mapGetters } from 'vuex'
export default {
    name: "Datasets",
    components: {
        Dataset: () => import('@/components/datasets/DatasetNew.vue')
    },
	data: () => ({
		selectedType: 1
	}),
	computed: {
		...mapGetters({
			datasets: 'datasets/getDatasets'
		})
	}
}
</script>

<style lang="scss" scoped>
.datasets-menu {
	width: 175px;
	border-right: 1px solid #0e1621;
	padding-top: 18px;
	hr {
		border: none;
		border-bottom: 1px solid #242F3D;
		margin: 20px;
	}
	&__items {
		&--item {
			padding: 10px 0 10px 20px;
			cursor: pointer;
			font-size: 14px;
			color: #A7BED3;
			display: block;
			* {
				vertical-align: middle;
			}
			i {
				margin-right: 20px;
			}
			&:hover {
				color: #65B9F4;
				.ci-world {
					background-color: #65B9F4;
				}
			}
			&.selected {
				background-color: #0E1621;
			}
		}
	}
	&__categories {
		font-size: 14px;
		padding-left: 20px;
		&--item {
			margin-bottom: 10px;
			li {
				padding-top: 5px;
				color: #6C7883;
				&:first-child {
					padding-top: 10px;
				}
			}
		}
	}
}

.cont {
	background-color: #17212B;
	padding: 0;
	display: flex;
	height: 100%;
}

.ci-icon {
	font-size: 20px;
}

.ci-world {
	display: inline-block;
	width: 20px;
	height: 20px;
	background-color: #A7BED3;
	mask-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTE3LjA3MSAxNy4wNzFDMTguOTU5NyAxNS4xODIzIDIwIDEyLjY3MSAyMCA5Ljk5OTg3QzIwIDcuMzI5IDE4Ljk1OTcgNC44MTc3MSAxNy4wNzEgMi45MjkwMUMxNS4xODIzIDEuMDQwMDUgMTIuNjcxIDAgOS45OTk4NyAwQzcuMzI5IDAgNC44MTc3MSAxLjA0MDA1IDIuOTI5MDEgMi45MjkwMUMxLjA0MDMxIDQuODE3NzEgMCA3LjMyOSAwIDkuOTk5ODdDMCAxMi42NzEgMS4wNDAwNSAxNS4xODIzIDIuOTI5MDEgMTcuMDcxQzQuODE3NzEgMTguOTU5NyA3LjMyODc0IDIwIDkuOTk5ODcgMjBDMTIuNjcxIDIwIDE1LjE4MjMgMTguOTU5NyAxNy4wNzEgMTcuMDcxWk04LjUwNjE0IDE2LjAwNzRDNy45ODkzNCAxNC42NTMxIDcuNjcxNDYgMTIuOTAyNSA3LjU4ODQzIDEwLjk5OThIMTIuNDExM0MxMi4zMjg1IDEyLjkwMjUgMTIuMDEwNyAxNC42NTMxIDExLjQ5MzkgMTYuMDA3NEMxMC45NzE0IDE3LjM3NjUgMTAuMzUzMiAxNy45OTk4IDkuOTk5ODcgMTcuOTk5OEM5LjY0NjU1IDE3Ljk5OTggOS4wMjgzNiAxNy4zNzY1IDguNTA2MTQgMTYuMDA3NFYxNi4wMDc0Wk05Ljk5OTg3IDEuOTk5OTJDMTAuMzUzMiAxLjk5OTkyIDEwLjk3MTQgMi42MjM1NCAxMS40OTM5IDMuOTkyNkMxMi4wMTA0IDUuMzQ2OTIgMTIuMzI4NSA3LjA5NzUgMTIuNDExMyA4Ljk5OTkxSDcuNTg4NDNDNy42NzEyIDcuMDk3NSA3Ljk4OTM0IDUuMzQ2OTIgOC41MDU4OCAzLjk5MjZDOS4wMjgzNiAyLjYyMzU0IDkuNjQ2NTUgMS45OTk5MiA5Ljk5OTg3IDEuOTk5OTJWMS45OTk5MlpNMTUuNjU2NyAxNS42NTY3QzE0LjkwNjggMTYuNDA2NSAxNC4wMzQxIDE2Ljk4OSAxMy4wODQzIDE3LjM4NThDMTMuODY3MyAxNS42OTkzIDE0LjMxNSAxMy4zODIgMTQuNDEzIDEwLjk5OThIMTcuOTM4M0MxNy43MjA1IDEyLjc1ODIgMTYuOTI5NSAxNC4zODQxIDE1LjY1NjcgMTUuNjU2N1YxNS42NTY3Wk0xNy45MzgzIDguOTk5OTFIMTQuNDEzQzE0LjMxNSA2LjYxNzY5IDEzLjg2NzMgNC4zMDA2NiAxMy4wODQzIDIuNjEzOTdDMTQuMDM0MSAzLjAxMDc1IDE0LjkwNjggMy41OTMyNCAxNS42NTY3IDQuMzQzMDhDMTYuOTI5NSA1LjYxNTkyIDE3LjcyMDUgNy4yNDE4MyAxNy45MzgzIDguOTk5OTFaTTQuMzQzMDggNC4zNDMwOEM1LjA5MjkyIDMuNTkzMjQgNS45NjU2MiAzLjAxMDc1IDYuOTE1NjcgMi42MTM5N0M2LjEzMjcyIDQuMzAwNCA1LjY4NDczIDYuNjE3NjkgNS41ODY2OSA4Ljk5OTkxSDIuMDYxNDhDMi4yNzkyNyA3LjI0MTgzIDMuMDcwNSA1LjYxNTkyIDQuMzQzMDggNC4zNDMwOFpNMi4wNjE0OCAxMC45OTk4SDUuNTg2NjlDNS42ODQ3MyAxMy4zODIgNi4xMzI3MiAxNS42OTkzIDYuOTE1NjcgMTcuMzg1OEM1Ljk2NTYyIDE2Ljk4OSA1LjA5MjkyIDE2LjQwNjggNC4zNDMwOCAxNS42NTY3QzMuMDcwNSAxNC4zODQxIDIuMjc5MjcgMTIuNzU4MiAyLjA2MTQ4IDEwLjk5OThWMTAuOTk5OFoiIGZpbGw9IiNBN0JFRDMiLz4KPC9zdmc+Cg==');
}
</style>